# Lint as: python3
# pylint: disable=g-bad-file-header,line-too-long
"""Training + small research add-ons (noise schedule + aux position loss).
"""
import collections
import functools
import json
import os
import pickle
from tqdm import tqdm

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tree

from learning_to_simulate import learned_simulator
from learning_to_simulate import noise_utils
from learning_to_simulate import reading_utils

# ----------------------- Flags -----------------------
flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'eval_rollout'],
                  'Train model, one-step evaluation, or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  'Split to use when running evaluation.')
flags.DEFINE_string('data_path', None, 'Dataset directory.')
flags.DEFINE_string('model_path', None, 'Checkpoint directory.')
flags.DEFINE_string('output_path', None, 'Output directory for rollouts.')
flags.DEFINE_integer('batch_size', 2, 'Batch size (graphs per batch).')
flags.DEFINE_integer('num_steps', 20000, 'Training steps (small by default).')

# original base noise std (used if schedule is constant)
flags.DEFINE_float('noise_std', 6.7e-4, 'Base noise std (used if constant).')

# NEW: noise schedule & aux position loss
flags.DEFINE_enum('noise_schedule', 'cosine', ['constant', 'linear', 'cosine'],
                  'Schedule for training-time input noise std.')
flags.DEFINE_float('noise_std_min', 3e-4, 'Min noise std for schedule.')
flags.DEFINE_float('noise_std_max', 1.2e-3, 'Max noise std for schedule.')
flags.DEFINE_float('aux_pos_loss_coef', 0.1,
                   'Coefficient for auxiliary one-step position MSE (>=0).')

# small model defaults for quick demo
flags.DEFINE_integer('latent_size', 64, 'Latent size for GNN.')
flags.DEFINE_integer('hidden_size', 64, 'Hidden size for MLPs.')
flags.DEFINE_integer('hidden_layers', 2, 'Number of hidden layers in MLPs.')
flags.DEFINE_integer('message_passing_steps', 5, 'Message passing steps.')

# rollout saving
flags.DEFINE_boolean('save_npz_rollout', True,
                     'Also save compact .npz for easy plotting.')
# StepCounterHook
flags.DEFINE_integer('steps_per_log', 500,
                     'Log steps/sec every N steps.')
flags.DEFINE_boolean('use_tqdm', True, 'Use tqdm progress bar during training.')
flags.DEFINE_integer('tqdm_update_every', 1, 'Update tqdm every N steps.')


FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

class TQDMHook(tf.train.SessionRunHook):
  """A simple tqdm progress bar for tf.estimator training."""
  def __init__(self, total_steps, every_n_steps=1, unit="step"):
    self.total_steps = int(total_steps)
    self.every_n_steps = int(every_n_steps)
    self.unit = unit
    self._last_gs = 0
    self._bar = None

  def begin(self):
    # graph tensor
    self._gs_tensor = tf.train.get_global_step()

  def after_create_session(self, session, coord):
    if tqdm is not None:
      self._bar = tqdm(total=self.total_steps, desc="train", unit=self.unit, ascii=True)
    self._last_gs = session.run(self._gs_tensor)

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=self._gs_tensor)

  def after_run(self, run_context, run_values):
    if self._bar is None:
      return
    gs = int(run_values.results)
    if (gs - self._last_gs) >= self.every_n_steps:
      delta = gs - self._last_gs
      self._bar.update(delta)
      self._last_gs = gs

  def end(self, session):
    if self._bar is not None:
      cur = self._bar.n
      if cur < self.total_steps:
        self._bar.update(self.total_steps - cur)
      self._bar.close()

# ----------------------- Data helpers -----------------------
def get_kinematic_mask(particle_types):
  return tf.equal(particle_types, KINEMATIC_PARTICLE_ID)

def prepare_inputs(tensor_dict):
  # position: [seq_len, n, dim] -> [n, seq_len, dim]
  pos = tensor_dict['position']
  pos = tf.transpose(pos, perm=[1, 0, 2])
  target_position = pos[:, -1]
  tensor_dict['position'] = pos[:, :-1]
  num_particles = tf.shape(pos)[0]
  tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]
  if 'step_context' in tensor_dict:
    tensor_dict['step_context'] = tensor_dict['step_context'][-2]
    tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
  return tensor_dict, target_position

def prepare_rollout_inputs(context, features):
  out_dict = {**context}
  pos = tf.transpose(features['position'], [1, 0, 2])
  target_position = pos[:, -1]
  out_dict['position'] = pos[:, :-1]
  out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
  if 'step_context' in features:
    out_dict['step_context'] = features['step_context']
  out_dict['is_trajectory'] = tf.constant([True], tf.bool)
  return out_dict, target_position

def batch_concat(dataset, batch_size):
  windowed_ds = dataset.window(batch_size)
  initial_state = tree.map_structure(
      lambda spec: tf.zeros(shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
      dataset.element_spec)
  def reduce_window(initial_state, ds):
    return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))
  return windowed_ds.map(lambda *x: tree.map_structure(reduce_window, initial_state, x))

def get_input_fn(data_path, batch_size, mode, split):
  def input_fn():
    metadata = _read_metadata(data_path)
    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
    ds = ds.map(functools.partial(
        reading_utils.parse_serialized_simulation_example, metadata=metadata))
    if mode.startswith('one_step'):
      split_with_window = functools.partial(reading_utils.split_trajectory,
                                            window_length=INPUT_SEQUENCE_LENGTH + 1)
      ds = ds.flat_map(split_with_window)
      ds = ds.map(prepare_inputs)
      if mode == 'one_step_train':
        ds = ds.repeat().shuffle(512)
      ds = batch_concat(ds, batch_size)
    elif mode == 'rollout':
      assert batch_size == 1
      ds = ds.map(prepare_rollout_inputs)
    else:
      raise ValueError(f'unknown mode: {mode}')
    return ds
  return input_fn


# ----------------------- Rollout -----------------------
def rollout(simulator, features, num_steps):
  initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]
  ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:]
  global_context = features.get('step_context')

  def step_fn(step, current_positions, predictions):
    if global_context is None:
      global_context_step = None
    else:
      global_context_step = global_context[step + INPUT_SEQUENCE_LENGTH - 1][tf.newaxis]
    next_position = simulator(current_positions,
                              n_particles_per_example=features['n_particles_per_example'],
                              particle_types=features['particle_type'],
                              global_context=global_context_step)
    # overwrite kinematic with GT
    kinematic_mask = get_kinematic_mask(features['particle_type'])
    next_gt = ground_truth_positions[:, step]
    next_position = tf.where(kinematic_mask, next_gt, next_position)
    predictions = predictions.write(step, next_position)
    next_positions = tf.concat([current_positions[:, 1:], next_position[:, tf.newaxis]], axis=1)
    return (step + 1, next_positions, predictions)

  predictions = tf.TensorArray(size=num_steps, dtype=tf.float32)
  _, _, predictions = tf.while_loop(
      cond=lambda step, *_: tf.less(step, num_steps),
      body=step_fn,
      loop_vars=(0, initial_positions, predictions),
      back_prop=False,
      parallel_iterations=1)

  out = {
      'initial_positions': tf.transpose(initial_positions, [1, 0, 2]),
      'predicted_rollout': predictions.stack(),
      'ground_truth_rollout': tf.transpose(ground_truth_positions, [1, 0, 2]),
      'particle_types': features['particle_type'],
  }
  if global_context is not None:
    out['global_context'] = global_context
  return out


# ----------------------- Simulator -----------------------
def _combine_std(std_x, std_y):
  return np.sqrt(std_x**2 + std_y**2)

def _get_simulator(model_kwargs, metadata, acc_noise_std, vel_noise_std):
  cast = lambda v: np.array(v, dtype=np.float32)
  acceleration_stats = Stats(cast(metadata['acc_mean']),
                             _combine_std(cast(metadata['acc_std']), acc_noise_std))
  velocity_stats = Stats(cast(metadata['vel_mean']),
                         _combine_std(cast(metadata['vel_std']), vel_noise_std))
  normalization_stats = {'acceleration': acceleration_stats, 'velocity': velocity_stats}
  if 'context_mean' in metadata:
    normalization_stats['context'] = Stats(cast(metadata['context_mean']),
                                           cast(metadata['context_std']))
  return learned_simulator.LearnedSimulator(
      num_dimensions=metadata['dim'],
      connectivity_radius=metadata['default_connectivity_radius'],
      graph_network_kwargs=model_kwargs,
      boundaries=metadata['bounds'],
      num_particle_types=NUM_PARTICLE_TYPES,
      normalization_stats=normalization_stats,
      particle_type_embedding_size=16)


# ----------------------- NEW: noise schedule -----------------------
def scheduled_noise_std(global_step, total_steps, sched, std_min, std_max):
  gs = tf.cast(global_step, tf.float32)
  ts = tf.cast(tf.maximum(total_steps, 1), tf.float32)
  p = tf.clip_by_value(gs / ts, 0.0, 1.0)
  if sched == 'constant':
    val = std_max
  elif sched == 'linear':
    val = std_min + (std_max - std_min) * p
  else:  # cosine
    val = std_min + (std_max - std_min) * 0.5 * (1.0 - tf.cos(np.pi * p))
  return val


# ----------------------- Estimator fns -----------------------
def get_one_step_estimator_fn(
    data_path,
    base_noise_std,
    latent_size=64,
    hidden_size=64,
    hidden_layers=2,
    message_passing_steps=5,
    noise_schedule='cosine',
    noise_std_min=3e-4,
    noise_std_max=1.2e-3,
    aux_pos_loss_coef=0.1,
    total_steps=20000):
  """One-step model with noise schedule + auxiliary position loss."""
  metadata = _read_metadata(data_path)
  model_kwargs = dict(
      latent_size=latent_size,
      mlp_hidden_size=hidden_size,
      mlp_num_hidden_layers=hidden_layers,
      num_message_passing_steps=message_passing_steps)

  def estimator_fn(features, labels, mode):
    target_next_position = labels
    global_step = tf.train.get_global_step()

    # scheduled noise for this step
    cur_noise_std = scheduled_noise_std(global_step, total_steps,
                                        noise_schedule, noise_std_min, noise_std_max)

    simulator = _get_simulator(model_kwargs, metadata,
                           acc_noise_std=noise_std_max,
                           vel_noise_std=noise_std_max)


    # random-walk noise on input sequence
    sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
        features['position'], noise_std_last_step=cur_noise_std)

    # mask out kinematic particles
    non_kinematic_mask = tf.logical_not(get_kinematic_mask(features['particle_type']))
    noise_mask = tf.cast(non_kinematic_mask, sampled_noise.dtype)[:, tf.newaxis, tf.newaxis]
    sampled_noise *= noise_mask

    # predicted & target normalized accelerations
    pred_accel, target_accel = simulator.get_predicted_and_target_normalized_accelerations(
        next_position=target_next_position,
        position_sequence=features['position'],
        position_sequence_noise=sampled_noise,
        n_particles_per_example=features['n_particles_per_example'],
        particle_types=features['particle_type'],
        global_context=features.get('step_context'))

    # accel loss (mask kinematic)
    accel_mse = (pred_accel - target_accel) ** 2
    accel_mse = tf.where(non_kinematic_mask, accel_mse, tf.zeros_like(accel_mse))
    denom = tf.reduce_sum(tf.cast(non_kinematic_mask, tf.float32))
    loss = tf.reduce_sum(accel_mse) / tf.maximum(denom, 1.0)

    # NEW: auxiliary one-step position MSE
    predicted_next_position = simulator(
        position_sequence=features['position'],
        n_particles_per_example=features['n_particles_per_example'],
        particle_types=features['particle_type'],
        global_context=features.get('step_context'))
    one_step_pos_mse = tf.reduce_mean((predicted_next_position - target_next_position) ** 2)
    if aux_pos_loss_coef > 0.0:
      loss = loss + aux_pos_loss_coef * one_step_pos_mse

    # LR schedule (same style as original)
    min_lr = 1e-6
    lr = tf.train.exponential_decay(learning_rate=1e-4 - min_lr,
                                    global_step=global_step,
                                    decay_steps=int(5e6),
                                    decay_rate=0.1) + min_lr
    train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step)

    eval_metrics_ops = {
        'mse_accel': tf.metrics.mean_squared_error(pred_accel, target_accel),
        'mse_pos_one_step': tf.metrics.mean_squared_error(predicted_next_position, target_next_position),
        'noise_std_current': tf.metrics.mean(cur_noise_std, cur_noise_std),
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss,
        predictions={'predicted_next_position': predicted_next_position},
        eval_metric_ops=eval_metrics_ops)

  return estimator_fn


def get_rollout_estimator_fn(data_path,
                             noise_std,
                             latent_size=64,
                             hidden_size=64,
                             hidden_layers=2,
                             message_passing_steps=5):
  metadata = _read_metadata(data_path)
  model_kwargs = dict(
      latent_size=latent_size,
      mlp_hidden_size=hidden_size,
      mlp_num_hidden_layers=hidden_layers,
      num_message_passing_steps=message_passing_steps)

  def estimator_fn(features, labels, mode):
    del labels
    simulator = _get_simulator(model_kwargs, metadata,
                               acc_noise_std=noise_std,
                               vel_noise_std=noise_std)
    num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
    roll = rollout(simulator, features, num_steps=num_steps)
    se = (roll['predicted_rollout'] - roll['ground_truth_rollout']) ** 2
    loss = tf.reduce_mean(se)
    eval_ops = {'rollout_error_mse': tf.metrics.mean_squared_error(
        roll['predicted_rollout'], roll['ground_truth_rollout'])}
    roll = tree.map_structure(lambda x: x[tf.newaxis], roll)
    return tf.estimator.EstimatorSpec(mode=mode, train_op=None, loss=loss,
                                      predictions=roll, eval_metric_ops=eval_ops)
  return estimator_fn


# ----------------------- IO -----------------------
def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())


# ----------------------- main -----------------------
def main(_):
  tf.disable_v2_behavior()
  # train / eval
  if FLAGS.mode in ['train', 'eval']:
    estimator = tf.estimator.Estimator(
        get_one_step_estimator_fn(
            data_path=FLAGS.data_path,
            base_noise_std=FLAGS.noise_std,
            latent_size=FLAGS.latent_size,
            hidden_size=FLAGS.hidden_size,
            hidden_layers=FLAGS.hidden_layers,
            message_passing_steps=FLAGS.message_passing_steps,
            noise_schedule=FLAGS.noise_schedule,
            noise_std_min=FLAGS.noise_std_min,
            noise_std_max=FLAGS.noise_std_max,
            aux_pos_loss_coef=FLAGS.aux_pos_loss_coef,
            total_steps=FLAGS.num_steps),
        model_dir=FLAGS.model_path)

    if FLAGS.mode == 'train':
        hooks = [tf.train.StepCounterHook(every_n_steps=FLAGS.steps_per_log)]
        if FLAGS.use_tqdm and tqdm is not None:
            hooks.append(TQDMHook(total_steps=FLAGS.num_steps,
                                every_n_steps=FLAGS.tqdm_update_every))
        estimator.train(
            input_fn=get_input_fn(FLAGS.data_path, FLAGS.batch_size,
                                mode='one_step_train', split='train'),
            max_steps=FLAGS.num_steps,
            hooks=hooks)


    else:
      eval_metrics = estimator.evaluate(
          input_fn=get_input_fn(FLAGS.data_path, FLAGS.batch_size,
                                mode='one_step', split=FLAGS.eval_split))
      logging.info('Evaluation metrics:')
      logging.info(eval_metrics)

  # rollout
  elif FLAGS.mode == 'eval_rollout':
    if not FLAGS.output_path:
      raise ValueError('A rollout path must be provided.')
    rollout_estimator = tf.estimator.Estimator(
        get_rollout_estimator_fn(
            FLAGS.data_path,
            FLAGS.noise_std,
            latent_size=FLAGS.latent_size,
            hidden_size=FLAGS.hidden_size,
            hidden_layers=FLAGS.hidden_layers,
            message_passing_steps=FLAGS.message_passing_steps),
        model_dir=FLAGS.model_path)

    metadata = _read_metadata(FLAGS.data_path)
    it = rollout_estimator.predict(
        input_fn=get_input_fn(FLAGS.data_path, batch_size=1,
                              mode='rollout', split=FLAGS.eval_split))

    if not os.path.exists(FLAGS.output_path):
      os.makedirs(FLAGS.output_path)

    for idx, ex in enumerate(it):
      ex['metadata'] = metadata
      pkl_name = os.path.join(FLAGS.output_path, f'rollout_{FLAGS.eval_split}_{idx}.pkl')
      logging.info('Saving: %s', pkl_name)
      with open(pkl_name, 'wb') as f:
        pickle.dump(ex, f)

      if FLAGS.save_npz_rollout:
        npz_name = os.path.join(FLAGS.output_path, f'rollout_{FLAGS.eval_split}_{idx}.npz')
        np.savez_compressed(
            npz_name,
            predicted_rollout=ex['predicted_rollout'],
            ground_truth_rollout=ex['ground_truth_rollout'],
            initial_positions=ex['initial_positions'],
            particle_types=ex['particle_types'])
        logging.info('Saved NPZ: %s', npz_name)

if __name__ == '__main__':
  app.run(main)
