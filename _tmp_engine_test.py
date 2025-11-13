from captionqa.captioning.panorama import PanoramaSamplingConfig
from captionqa.captioning.config import QwenVLConfig
from captionqa.captioning.engines.qwen_vl import QwenVLEngine

engine = QwenVLEngine.from_configs(sampler_cfg=PanoramaSamplingConfig(), qwen_cfg=QwenVLConfig())
processor, model = engine._load_model()
print(bool(processor), bool(model))
