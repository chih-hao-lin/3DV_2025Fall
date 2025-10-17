# Examples closer to rigid body
python train.py --config=configs/elastic/8.py
python test.py --config=configs/elastic/8.py --num-frame=48 --cam-id=0

# python train.py --config=configs/elastic/8_biased.py
# python test.py --config=configs/elastic/8_biased.py --num-frame=48 --cam-id=0

python train.py --config=configs/elastic/8_rigid.py
python test.py --config=configs/elastic/8_rigid.py --num-frame=48 --cam-id=0

# Examples closer to deformable body
python train.py --config=configs/elastic/0.py
python test.py --config=configs/elastic/0.py --num-frame=48 --cam-id=0

python train.py --config=configs/elastic/0_rigid.py
python test.py --config=configs/elastic/0_rigid.py --num-frame=48 --cam-id=0
