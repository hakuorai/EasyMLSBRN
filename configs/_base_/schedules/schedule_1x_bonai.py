optimizer = dict(
    type="SGD",
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001,
)
optimizer_config = dict(
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    ),
)
# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],
)
runner = dict(
    type="StagedEpochBasedRunner",
    max_epochs=12,
    supervised_epochs=0,
)
