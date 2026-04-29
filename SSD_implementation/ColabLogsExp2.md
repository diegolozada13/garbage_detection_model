
>>> Testing: LR=0.01, WD=0.0005
Downloading: "https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth" to /root/.cache/torch/hub/checkpoints/ssd300_vgg16_coco-b556d3b4.pth
100%|██████████| 136M/136M [00:00<00:00, 174MB/s]
/tmp/ipykernel_5519/751652916.py:15: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = GradScaler()
Training:   0%|          | 0/37 [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/tmp/ipykernel_5519/2836796855.py:8: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
Training: 100%|██████████| 37/37 [02:22<00:00,  3.85s/it]
  Ep 1: Train=nan, Val=8900.5306, LR=0.010000
Training: 100%|██████████| 37/37 [00:23<00:00,  1.57it/s]
  Ep 2: Train=nan, Val=8900.5306, LR=0.010000
Training: 100%|██████████| 37/37 [00:25<00:00,  1.44it/s]
  Ep 3: Train=nan, Val=8900.5306, LR=0.010000
Training: 100%|██████████| 37/37 [00:24<00:00,  1.51it/s]
  Ep 4: Train=nan, Val=8900.5306, LR=0.010000
  Early stopping config...

>>> Testing: LR=0.01, WD=0.0001
Training: 100%|██████████| 37/37 [00:24<00:00,  1.53it/s]
  Ep 1: Train=nan, Val=434129602169555654803456.0000, LR=0.010000
Training: 100%|██████████| 37/37 [00:23<00:00,  1.58it/s]
  Ep 2: Train=nan, Val=434129602169555654803456.0000, LR=0.010000
Training: 100%|██████████| 37/37 [00:23<00:00,  1.58it/s]
  Ep 3: Train=nan, Val=434129602169555654803456.0000, LR=0.010000
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
  Ep 4: Train=nan, Val=434129602169555654803456.0000, LR=0.010000
  Early stopping config...

>>> Testing: LR=0.001, WD=0.0005
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
  Ep 1: Train=12.1426, Val=8.8231, LR=0.001000
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
  Ep 2: Train=7.1316, Val=6.6446, LR=0.001000
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
  Ep 3: Train=5.8743, Val=6.1097, LR=0.001000
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
  Ep 4: Train=5.4293, Val=5.8612, LR=0.001000
Training: 100%|██████████| 37/37 [00:24<00:00,  1.50it/s]
  Ep 5: Train=5.2165, Val=5.7322, LR=0.000100
Training: 100%|██████████| 37/37 [00:25<00:00,  1.44it/s]
  Ep 6: Train=4.9953, Val=5.6500, LR=0.000100
Training: 100%|██████████| 37/37 [00:24<00:00,  1.48it/s]
  Ep 7: Train=4.9380, Val=5.6305, LR=0.000100
Training: 100%|██████████| 37/37 [00:23<00:00,  1.54it/s]
  Ep 8: Train=4.9293, Val=5.6186, LR=0.000010
Training: 100%|██████████| 37/37 [00:25<00:00,  1.45it/s]
  Ep 9: Train=4.8812, Val=5.6142, LR=0.000010
Training: 100%|██████████| 37/37 [00:24<00:00,  1.53it/s]
  Ep 10: Train=4.8816, Val=5.6106, LR=0.000010

>>> Testing: LR=0.001, WD=0.0001
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
  Ep 1: Train=12.3578, Val=8.5899, LR=0.001000
Training: 100%|██████████| 37/37 [00:23<00:00,  1.57it/s]
  Ep 2: Train=6.7473, Val=6.3944, LR=0.001000
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
  Ep 3: Train=5.6183, Val=5.9590, LR=0.001000
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
  Ep 4: Train=5.2832, Val=5.7671, LR=0.001000
Training: 100%|██████████| 37/37 [00:23<00:00,  1.59it/s]
  Ep 5: Train=5.0476, Val=5.7855, LR=0.000100
Training: 100%|██████████| 37/37 [00:23<00:00,  1.59it/s]
  Ep 6: Train=4.8634, Val=5.5718, LR=0.000100
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
  Ep 7: Train=4.8225, Val=5.5535, LR=0.000100
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
  Ep 8: Train=4.8360, Val=5.5672, LR=0.000010
Training: 100%|██████████| 37/37 [00:25<00:00,  1.48it/s]
  Ep 9: Train=4.8073, Val=5.5559, LR=0.000010
Training: 100%|██████████| 37/37 [00:25<00:00,  1.48it/s]
  Ep 10: Train=4.7866, Val=5.5501, LR=0.000010

>>> Testing: LR=0.0005, WD=0.0005
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
  Ep 1: Train=12.7562, Val=9.2264, LR=0.000500
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
  Ep 2: Train=7.5396, Val=6.8822, LR=0.000500
Training: 100%|██████████| 37/37 [00:24<00:00,  1.52it/s]
  Ep 3: Train=5.9296, Val=6.2010, LR=0.000500
Training: 100%|██████████| 37/37 [00:24<00:00,  1.51it/s]
  Ep 4: Train=5.4649, Val=5.9239, LR=0.000500
Training: 100%|██████████| 37/37 [00:24<00:00,  1.50it/s]
  Ep 5: Train=5.2189, Val=5.7804, LR=0.000050
Training: 100%|██████████| 37/37 [00:24<00:00,  1.51it/s]
  Ep 6: Train=5.1227, Val=5.7185, LR=0.000050
Training: 100%|██████████| 37/37 [00:24<00:00,  1.49it/s]
  Ep 7: Train=5.0552, Val=5.7007, LR=0.000050
Training: 100%|██████████| 37/37 [00:24<00:00,  1.51it/s]
  Ep 8: Train=5.0441, Val=5.6815, LR=0.000005
Training: 100%|██████████| 37/37 [00:24<00:00,  1.53it/s]
  Ep 9: Train=5.0100, Val=5.6823, LR=0.000005
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
  Ep 10: Train=5.0330, Val=5.6806, LR=0.000005

>>> Testing: LR=0.0005, WD=0.0001
Training: 100%|██████████| 37/37 [00:24<00:00,  1.53it/s]
  Ep 1: Train=12.8691, Val=9.4746, LR=0.000500
Training: 100%|██████████| 37/37 [00:24<00:00,  1.54it/s]
  Ep 2: Train=7.6435, Val=7.0845, LR=0.000500
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
  Ep 3: Train=6.0332, Val=6.2021, LR=0.000500
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
  Ep 4: Train=5.5209, Val=5.9380, LR=0.000500
Training: 100%|██████████| 37/37 [00:24<00:00,  1.54it/s]
  Ep 5: Train=5.2643, Val=5.8049, LR=0.000050
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
  Ep 6: Train=5.1124, Val=5.7847, LR=0.000050
Training: 100%|██████████| 37/37 [00:24<00:00,  1.52it/s]
  Ep 7: Train=5.0811, Val=5.7328, LR=0.000050
Training: 100%|██████████| 37/37 [00:24<00:00,  1.52it/s]
  Ep 8: Train=5.0506, Val=5.7239, LR=0.000005
Training: 100%|██████████| 37/37 [00:24<00:00,  1.53it/s]
  Ep 9: Train=4.9954, Val=5.7219, LR=0.000005
Training: 100%|██████████| 37/37 [00:24<00:00,  1.53it/s]
  Ep 10: Train=5.0456, Val=5.7204, LR=0.000005

FINISH TUNING. Best Params: {'lr': 0.001, 'wd': 0.0001} (Loss: 5.5501)


Starting Final Training with LR=0.001, WD=0.0001...
/tmp/ipykernel_5519/1950379842.py:10: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = GradScaler()
Training:   0%|          | 0/37 [00:00<?, ?it/s]/tmp/ipykernel_5519/2836796855.py:8: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
Training: 100%|██████████| 37/37 [00:24<00:00,  1.53it/s]
Epoch 1/50 - Loss: 12.9550, Val: 9.0358
  *** New Best Model Saved! Loss: 9.0358 ***
Training: 100%|██████████| 37/37 [00:24<00:00,  1.53it/s]
Epoch 2/50 - Loss: 7.2169, Val: 6.6547
  *** New Best Model Saved! Loss: 6.6547 ***
Training: 100%|██████████| 37/37 [00:25<00:00,  1.45it/s]
Epoch 3/50 - Loss: 5.8260, Val: 5.9989
  *** New Best Model Saved! Loss: 5.9989 ***
Training: 100%|██████████| 37/37 [00:25<00:00,  1.47it/s]
Epoch 4/50 - Loss: 5.3865, Val: 5.7670
  *** New Best Model Saved! Loss: 5.7670 ***
Training: 100%|██████████| 37/37 [00:25<00:00,  1.47it/s]
Epoch 5/50 - Loss: 5.2068, Val: 5.6980
  *** New Best Model Saved! Loss: 5.6980 ***
Training: 100%|██████████| 37/37 [00:25<00:00,  1.46it/s]
Epoch 6/50 - Loss: 4.9724, Val: 5.6535
  *** New Best Model Saved! Loss: 5.6535 ***
Training: 100%|██████████| 37/37 [00:24<00:00,  1.48it/s]
Epoch 7/50 - Loss: 4.8699, Val: 5.5694
  *** New Best Model Saved! Loss: 5.5694 ***
Training: 100%|██████████| 37/37 [00:24<00:00,  1.48it/s]
Epoch 8/50 - Loss: 4.7476, Val: 5.5280
  *** New Best Model Saved! Loss: 5.5280 ***
Training: 100%|██████████| 37/37 [00:25<00:00,  1.46it/s]
Epoch 9/50 - Loss: 4.6684, Val: 5.5226
  *** New Best Model Saved! Loss: 5.5226 ***
Training: 100%|██████████| 37/37 [00:25<00:00,  1.44it/s]
Epoch 10/50 - Loss: 4.5640, Val: 5.4644
  *** New Best Model Saved! Loss: 5.4644 ***
Training: 100%|██████████| 37/37 [00:25<00:00,  1.46it/s]
Epoch 11/50 - Loss: 4.4969, Val: 5.5088
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
Epoch 12/50 - Loss: 4.4181, Val: 5.4945
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
Epoch 13/50 - Loss: 4.3451, Val: 5.4331
  *** New Best Model Saved! Loss: 5.4331 ***
Training: 100%|██████████| 37/37 [00:24<00:00,  1.50it/s]
Epoch 14/50 - Loss: 4.2915, Val: 5.3703
  *** New Best Model Saved! Loss: 5.3703 ***
Training: 100%|██████████| 37/37 [00:24<00:00,  1.48it/s]
Epoch 15/50 - Loss: 4.2864, Val: 5.3959
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
Epoch 16/50 - Loss: 4.1284, Val: 5.4179
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
Epoch 17/50 - Loss: 4.1501, Val: 5.3374
  *** New Best Model Saved! Loss: 5.3374 ***
Training: 100%|██████████| 37/37 [00:25<00:00,  1.45it/s]
Epoch 18/50 - Loss: 4.0815, Val: 5.3060
  *** New Best Model Saved! Loss: 5.3060 ***
Training: 100%|██████████| 37/37 [00:25<00:00,  1.47it/s]
Epoch 19/50 - Loss: 4.0617, Val: 5.3898
Training: 100%|██████████| 37/37 [00:24<00:00,  1.54it/s]
Epoch 20/50 - Loss: 3.9743, Val: 5.3547
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
Epoch 21/50 - Loss: 3.9086, Val: 5.3680
Training: 100%|██████████| 37/37 [00:23<00:00,  1.54it/s]
Epoch 22/50 - Loss: 3.8995, Val: 5.3741
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
Epoch 23/50 - Loss: 3.8352, Val: 5.3359
Training: 100%|██████████| 37/37 [00:24<00:00,  1.54it/s]
Epoch 24/50 - Loss: 3.7790, Val: 5.4495
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
Epoch 25/50 - Loss: 3.7204, Val: 5.5028
Training: 100%|██████████| 37/37 [00:23<00:00,  1.58it/s]
Epoch 26/50 - Loss: 3.7424, Val: 5.3453
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
Epoch 27/50 - Loss: 3.6991, Val: 5.4996
Training: 100%|██████████| 37/37 [00:24<00:00,  1.53it/s]
Epoch 28/50 - Loss: 3.6731, Val: 5.4345
Training: 100%|██████████| 37/37 [00:24<00:00,  1.50it/s]
Epoch 29/50 - Loss: 3.5991, Val: 5.3887
Training: 100%|██████████| 37/37 [00:24<00:00,  1.49it/s]
Epoch 30/50 - Loss: 3.5429, Val: 5.4803
Training: 100%|██████████| 37/37 [00:24<00:00,  1.49it/s]
Epoch 31/50 - Loss: 3.4663, Val: 5.4013
Training: 100%|██████████| 37/37 [00:25<00:00,  1.46it/s]
Epoch 32/50 - Loss: 3.3665, Val: 5.3846
Training: 100%|██████████| 37/37 [00:25<00:00,  1.47it/s]
Epoch 33/50 - Loss: 3.3350, Val: 5.3960
Training: 100%|██████████| 37/37 [00:24<00:00,  1.48it/s]
Epoch 34/50 - Loss: 3.3447, Val: 5.3749
Training: 100%|██████████| 37/37 [00:24<00:00,  1.49it/s]
Epoch 35/50 - Loss: 3.3535, Val: 5.3542
Training: 100%|██████████| 37/37 [00:24<00:00,  1.51it/s]
Epoch 36/50 - Loss: 3.3284, Val: 5.4032
Training: 100%|██████████| 37/37 [00:24<00:00,  1.53it/s]
Epoch 37/50 - Loss: 3.3349, Val: 5.3817
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
Epoch 38/50 - Loss: 3.3334, Val: 5.3673
Training: 100%|██████████| 37/37 [00:24<00:00,  1.54it/s]
Epoch 39/50 - Loss: 3.3104, Val: 5.3568
Training: 100%|██████████| 37/37 [00:23<00:00,  1.56it/s]
Epoch 40/50 - Loss: 3.3192, Val: 5.3830
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
Epoch 41/50 - Loss: 3.2664, Val: 5.3791
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
Epoch 42/50 - Loss: 3.2561, Val: 5.3768
Training: 100%|██████████| 37/37 [00:23<00:00,  1.55it/s]
Epoch 43/50 - Loss: 3.2800, Val: 5.3749
Training: 100%|██████████| 37/37 [00:24<00:00,  1.54it/s]
Epoch 44/50 - Loss: 3.2973, Val: 5.3726
Training: 100%|██████████| 37/37 [00:24<00:00,  1.50it/s]
Epoch 45/50 - Loss: 3.2828, Val: 5.3716
Training: 100%|██████████| 37/37 [00:25<00:00,  1.47it/s]
Epoch 46/50 - Loss: 3.2557, Val: 5.3735
Training: 100%|██████████| 37/37 [00:25<00:00,  1.47it/s]
Epoch 47/50 - Loss: 3.2695, Val: 5.3757
Training: 100%|██████████| 37/37 [00:25<00:00,  1.48it/s]
Epoch 48/50 - Loss: 3.3066, Val: 5.3783
Training: 100%|██████████| 37/37 [00:25<00:00,  1.47it/s]
Epoch 49/50 - Loss: 3.2482, Val: 5.3747
Training: 100%|██████████| 37/37 [00:25<00:00,  1.48it/s]
Epoch 50/50 - Loss: 3.2787, Val: 5.3804
