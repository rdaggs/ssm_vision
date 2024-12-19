

Use:

``` python
model = Vim(dim=256,
            dt_rank=32,
            dim_inner=256,
            d_state=256,
            num_classes=10,
            image_size=32,
            patch_size=16,
            channels=3,
            dropout=0.1,
            depth=10,)
```

to load the following checkpoints:

-
-
-



Use:

``` python
model = Vim(dim=96,
            dt_rank=16,
            dim_inner=96,
            d_state=96,
            num_classes=10,
            image_size=32,
            patch_size=4,
            channels=3,
            dropout=0.1,
            depth=10,)
```

to load the following checkpoints:

- **SE_vim.pt** (Squeeze and Excitation integrated)
- **MobileNetV2_vim.pt** (MobileNetV2 integrated)
- **DynamicPrunning_vim** (Dynamic Prunning with Cosine Scheduler integrated)