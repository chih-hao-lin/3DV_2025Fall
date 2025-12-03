# Improve Diamond Performance for BankHeist

## Hacker: Qinjun Jiang (qinjunj2)

## Set up
Same as [Diamond](https://github.com/eloialonso/diamond).

## What Did I Do?

- Fine-tuned the model for an additional 20k steps on BankHeist, but observed no performance improvement. The trained model can be found in `diamond/outputs/`.

- Investigated action patterns and reward traces in BankHeist and identified the issue with purely random actions changing every step. The traces can be found in `diamond/dataset/rec_test*.txt/`. 

- Added support for action repetition (no-ops) between action changes

    - Usage: `--action-repeat <#>` Repeat the same action for # times before changing.
    ```bash
    python src/play.py --action-repeat 8
    ```

    - Result: more visually stable behavior.

- Experimented with replacing the history representation (concatenated frames + actions) with a latent-state encoder.

    - Trained from scratch for 10k steps, reaching performance comparable to the original Diamond model.

## Acknowledgement

This code is modified on top of [Diamond](https://github.com/eloialonso/diamond). Thanks for their impressive works.
