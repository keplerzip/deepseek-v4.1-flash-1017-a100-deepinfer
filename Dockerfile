FROM deepseek-v4.1-flash-1017-a100-deepinfer:20261113-r8-pr66497-8005
COPY source/r9-overlay/ /opt/deepinfer/source/r9-overlay/
COPY scripts/install_r9_overlay.py /opt/deepinfer/scripts/install_r9_overlay.py
RUN /opt/r8-venv/bin/python /opt/deepinfer/scripts/install_r9_overlay.py
COPY *.sh /opt/deepinfer/
COPY adapter/ /opt/deepinfer/adapter/
COPY scripts/ /opt/deepinfer/scripts/
COPY tests/ /opt/deepinfer/tests/
COPY configs/ /opt/deepinfer/configs/
COPY LICENSE NOTICE VERSION /opt/deepinfer/
RUN /opt/r8-venv/bin/python -m compileall -q -f /opt/deepinfer/adapter /opt/deepinfer/scripts
LABEL deepinfer.release="20261115-R9-EXPERIMENTAL" deepinfer.base="R8-HF2"
