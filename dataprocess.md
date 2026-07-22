# MCAP 目录转换

## 0. 启动 Matrix Studio 容器

以下命令在宿主机执行。当前使用的 Matrix Studio 镜像是：

```text
imagepublic.genrobotai.com/genrobot/matrix-studio:0.2.15
```

当前用户如果没有 Docker 权限，需要使用 `sudo`：

```bash
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy

sudo ./start_studio_sdk.sh \
  --image-name imagepublic.genrobotai.com/genrobot/matrix-studio:0.2.15
```

启动脚本会把宿主机目录：

```text
/home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/data
```

挂载到容器目录：

```text
/app/data
```

进入容器后，终端提示符通常类似 `root@容器ID:/app#`。确认 CQ 数据和
VIO 脚本都能访问：

```bash
ls /app/data/CQ | head
ls /app/scripts/process_mcap_inner.sh
```

如果容器已经存在但处于停止状态，可以在宿主机执行：

```bash
sudo docker start -ai matrix-studio
```

## 1. 批量生成 VIO

启动 Matrix Studio 容器后，在容器中执行：

```bash
mkdir -p /app/data/vio_output_cq0714

total=$(find /app/data/20260714/cq0714 -maxdepth 1 -name '*.mcap' | wc -l)
i=0
for f in /app/data/20260714/cq0714/*.mcap; do
  i=$((i + 1))
  echo "[$i/$total] $f"
  bash /app/scripts/process_mcap_inner.sh "$f" \
    --output-dir /app/data/vio_output_cq0714
done
```

## 2. 查看转换进度

在宿主机的另一个终端执行：

```bash
watch -n 2 'echo -n "已完成/总数: "; find /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/data/vio_output_cq0714 -maxdepth 1 -name "*_vio.mcap" | wc -l | tr "\n" "/"; find /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/data/20260714/cq0714 -maxdepth 1 -name "*.mcap" | wc -l'
```

## 3. 转成训练数据

回到宿主机执行：

```bash
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-diffusion-training
conda activate umi_mcap

mkdir -p data/tron1_gripper_20260714

python utils/mcap_to_zarr.py \
  /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/data/vio_output_cq0714 \
  -o data/tron1_gripper_20260714/dataset_raw.npz \
  --image-size 224,224
```

### 多个目录合并生成一个 NPZ

`mcap_to_zarr.py` 支持在 `-o` 前传入多个目录。下面的命令会将
`vio_output_cq0714` 和 `vio_output_jc0714` 中的所有 `.mcap` 合并为一个
训练数据集，每个 `.mcap` 对应一个 episode：

```bash
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-diffusion-training
conda activate umi_mcap

mkdir -p data/tron1_gripper_20260714

python utils/mcap_to_zarr.py \
  /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/data/vio_output_cq0714 \
  /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/data/vio_output_jc0714 \
  -o data/tron1_gripper_20260714/dataset_raw_cq_jc.npz \
  --image-size 224,224
```

生成文件：

```text
/home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-diffusion-training/data/tron1_gripper_20260714/dataset_raw_cq_jc.npz
```

## 4. 上传 NPZ 到远端服务器

在本机执行。远端的 `/home/ubuntu/projects` 是指向
`/data/chenqian/projects` 的软连接，因此文件实际保存在 `/data` 盘：

```bash
ssh ubuntu@117.50.75.55 \
  'mkdir -p /home/ubuntu/projects/arx-difussion-deploy/umi-diffusion-training/data/tron1_gripper_20260714'

scp /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-diffusion-training/data/tron1_gripper_20260714/dataset_raw_cq_jc.npz \
  ubuntu@117.50.75.55:/home/ubuntu/projects/arx-difussion-deploy/umi-diffusion-training/data/tron1_gripper_20260714/
```

上传后检查远端文件：

```bash
ssh ubuntu@117.50.75.55 \
  'ls -lh /home/ubuntu/projects/arx-difussion-deploy/umi-diffusion-training/data/tron1_gripper_20260714/dataset_raw_cq_jc.npz'
```

## 5. 从远端下载训练好的 CKPT 和 YAML

先在远端查找最近生成的 checkpoint。以下命令在本机执行：

```bash
ssh ubuntu@117.50.75.55 \
  'find /home/ubuntu/projects/arx-difussion-deploy/umi-diffusion-training/data/outputs -path "*/checkpoints/*.ckpt" -printf "%T@ %p\n" | sort -nr | head -20'
```

确定训练运行目录后设置 `REMOTE_RUN_DIR`。它应当是 `checkpoints` 和
`.hydra` 的上一级目录，例如：

```bash
REMOTE_RUN_DIR=/home/ubuntu/projects/arx-difussion-deploy/umi-diffusion-training/data/outputs/2026.07.14/23.23.50_train_diffusion_unet_timm_umi
LOCAL_MODEL_DIR=/home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714

mkdir -p "$LOCAL_MODEL_DIR"

scp "ubuntu@117.50.75.55:${REMOTE_RUN_DIR}/checkpoints/latest.ckpt" \
  "$LOCAL_MODEL_DIR/latest.ckpt"

scp "ubuntu@117.50.75.55:${REMOTE_RUN_DIR}/.hydra/config.yaml" \
  "$LOCAL_MODEL_DIR/latest.yaml"
```

检查下载结果：

```bash
ls -lh "$LOCAL_MODEL_DIR/latest.ckpt" "$LOCAL_MODEL_DIR/latest.yaml"
```

部署推理时使用：

```text
-i /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.ckpt
-c /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.yaml
```
