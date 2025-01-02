export const HEARTBEAT_INTERVAL = 5000; // 5 seconds

export const PRIMARY_WIDGETS = {
    CLIPSetLastLayer: "stop_at_clip_layer",
    CLIPTextEncode: "text",
    VAELoader: "vae_name",
    TomePatchModel: "ratio",
    SaveImage: "filename_prefix",
    LoadImage: "image"
};

export const RENAME_DEFAULTS = [
    'prompt', 'promptneg',
    'cfg', 'chg', 'img',
    'ccg{ControlNetApply}', 'cn_img{LoadImage}'
];

export const NODE_COLORS = {
    loader: [0, 0.4, 0.3],
    clip: [20, 0.4, 0.3],
    note: [40, 0.4, 0.3],
    sampler: [60, 0.4, 0.3],
    controlnet: [80, 0.4, 0.3],
    vae: [100, 0.4, 0.3],
    conditioning: [120, 0.4, 0.3],
    latent: [140, 0.4, 0.3],
    mask: [160, 0.4, 0.3],
    image: [180, 0.4, 0.3],
    style: [200, 0.4, 0.3],
    primitive: [220, 0.4, 0.3],
    gligen: [240, 0.4, 0.3],
}; 