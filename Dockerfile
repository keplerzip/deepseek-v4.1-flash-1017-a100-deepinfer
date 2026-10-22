FROM deepseek-v4.1-flash-1017-a100:20261018-r2
COPY vendor/wheels/ /opt/deepinfer/vendor/wheels/
COPY vendor/requirements.lock /opt/deepinfer/vendor/requirements.lock
RUN /usr/bin/python3 -m pip install --break-system-packages --no-index --no-deps --require-hashes --find-links=/opt/deepinfer/vendor/wheels -r /opt/deepinfer/vendor/requirements.lock
COPY LICENSE NOTICE /opt/deepinfer/
COPY vendor/RECIPE-LICENSE vendor/RECIPE-THIRD-PARTY-LICENSES /opt/deepinfer/vendor/
COPY adapter/ /opt/deepinfer/adapter/
COPY scripts/ /opt/deepinfer/scripts/
COPY *.sh /opt/deepinfer/
COPY source/ /opt/deepinfer/source/
COPY configs/ /opt/deepinfer/configs/
COPY vendor/sm80/ /opt/deepinfer/vendor/sm80/
COPY vendor/tokenizer/ /opt/deepinfer/vendor/tokenizer/
COPY vendor/r2-source/ /opt/deepinfer/vendor/r2-source/
COPY vendor/official-recipe/ /opt/deepinfer/vendor/official-recipe/
COPY vendor/model-manifest.json /opt/deepinfer/vendor/model-manifest.json
COPY tests/ /opt/deepinfer/tests/
RUN /usr/bin/python3 /opt/deepinfer/scripts/install_overlay.py && /usr/bin/python3 /opt/deepinfer/scripts/install_beta_overlay.py
ENV PYTHONPATH=/opt/deepinfer HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    PYTHONDONTWRITEBYTECODE=1 DEEPINFER_TOKENIZER=/opt/deepinfer/vendor/tokenizer/tokenizer.json \
    DEEPINFER_R2_SOURCE=/opt/deepinfer/vendor/r2-source \
    VLLM_DSV41_DEEPINFER_FUSED_COMBINE=1 VLLM_DSV41_DEEPINFER_MHC_ALIGNED=1 \
    DEEPINFER_BETA_UNIFIED_MLA=1 DEEPINFER_BETA_ENGRAM_PREFETCH=1 DEEPINFER_BETA_YOCO_FFN=0 DEEPINFER_BETA_YOCO_KEEP=256 \
    VLLM_USE_SIMPLE_KV_OFFLOAD=0 VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
LABEL org.opencontainers.image.title="DeepSeek 1017 A100 SM80 formal CPU KV runtime with official recipe server" \
      deepinfer.base-image="sha256:a63405a72ce202bb8347f17fc2432be0df484b794fba0224fd210b6daff004f6" \
      deepinfer.recipe-version="0.2.0"
ENTRYPOINT ["/usr/bin/python3", "/opt/deepinfer/scripts/launch.py"]
