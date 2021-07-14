# Swin-Transformer-Serve

Deploy Pre-trained [Swin-Transformer](https://github.com/microsoft/Swin-Transformer/) classifier trained on ImageNet 1K using [TorchServe](https://github.com/pytorch/serve)

### Create and activate conda env
```bash
conda create -n STS python=3.7 -y
conda activate STS
```

### Clone and Install dependencies
```bash
git clone https://github.com/kamalkraj/Swin-Transformer-Serve.git
cd Swin-Transformer-Serve
# clone TorchServe
git clone https://github.com/pytorch/serve.git
cd serve
# The instruction below is for CPU
python ./ts_scripts/install_dependencies.py
# The instruction below is for GPU
python ./ts_scripts/install_dependencies.py --cuda=cu102
#For conda
conda install torchserve torch-model-archiver torch-workflow-archiver -c pytorch
#For pip
pip install torchserve torch-model-archiver torch-workflow-archiver
```

### Pretrained Weights and Config

#### Download weights.
| name   | pretrain     | resolution | acc@1 | acc@5 | 1K model                                                                                                                                                                          |
|--------|--------------|------------|-------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Swin-T | ImageNet-1K  | 224x224    | 81.2  | 95.5  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)/[baidu](https://pan.baidu.com/s/156nWJy4Q28rDlrX-rRbI3w)           |
| Swin-S | ImageNet-1K  | 224x224    | 83.2  | 96.2  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth)/[baidu](https://pan.baidu.com/s/1KFjpj3Efey3LmtE1QqPeQg)          |
| Swin-B | ImageNet-1K  | 224x224    | 83.5  | 96.5  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth)/[baidu](https://pan.baidu.com/s/16bqCTEc70nC_isSsgBSaqQ)           |
| Swin-B | ImageNet-1K  | 384x384    | 84.5  | 97.0  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth)/[baidu](https://pan.baidu.com/s/1xT1cu740-ejW7htUdVLnmw)          |
| Swin-B | ImageNet-22K | 224x224    | 85.2  | 97.5  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1n_wNkcbRxVXit8r_KrfAVg)   |
| Swin-B | ImageNet-22K | 384x384    | 86.4  | 98.0  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1caKTSdoLJYoi4WBcnmWuWg)  |
| Swin-L | ImageNet-22K | 224x224    | 86.3  | 97.9  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1NkQApMWUhxBGjk1ne6VqBQ)  |
| Swin-L | ImageNet-22K | 384x384    | 87.3  | 98.2  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1X0FLHQyPOC6Kmv2CmgxJvA) |


For demo we will use Tiny model.
```bash
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth -O weights/swin_tiny_patch4_window7_224.pth
```

Copy the corresonding model config file from to `swin_config.yaml` 
```bash
cp configs/swin_tiny_patch4_window7_224.yaml swin_config.yaml
```

### TorchServe

Create MAR file using [torch-model-archiver](https://github.com/pytorch/serve/tree/master/model-archiver).

```bash
torch-model-archiver --model-name swin -v 1.0 --model-file swin_transformer.py --serialized-file weights/swin_tiny_patch4_window7_224.pth --handler swin_handler.py --extra-files index_to_name.json,swin_config.yaml --requirements-file requirements.txt
# move the swin.mar to model_store folder
mv swin.mar model_store
```

Start the `torchserve` using below cmd

in the config.properties we have set the configurations.[Refer](https://github.com/pytorch/serve/blob/master/docs/configuration.md)
```bash
torchserve --start --ncs --model-store model_store --models swin=swin.mar
```
Use ping API to check the health status of a running TorchServe server:
```bash
curl http://127.0.0.1:8080/ping
```
If the server is running, the response is:
```bash
{
  "health": "healthy!"
}
```
test prediction using cURL
```bash
curl http://127.0.0.1:8080/predictions/swin -T kitten_small.jpg
```
```bash
{
  "tabby": 0.44951513409614563,
  "tiger_cat": 0.1962115466594696,
  "lynx": 0.16013166308403015,
  "Egyptian_cat": 0.08244507014751434,
  "tiger": 0.015334611758589745
}
```

As we set in the configuration file only one worker will be created per model.
To increase workers and batch inference can done using [Management API](https://github.com/pytorch/serve/blob/master/docs/management_api.md)

#### Increase worker
```bash
curl -v -X PUT "http://localhost:8081/models/swin?min_worker=3"
```
This will increase minimum workers to 3.

#### Batch Inference.
[Reference](https://github.com/pytorch/serve/blob/master/docs/batch_inference_with_ts.md)
```bash
#if the model is already running dergister it first
curl -X DELETE "http://localhost:8081/models/swin/1.0"
# The following command will register a model "swin.mar" and configure TorchServe to use a batch_size of 8 and a max batch delay of 50 milli seconds.
curl -X POST "localhost:8081/models?url=swin.mar&batch_size=8&max_batch_delay=50"
```

Stop torchserve
```bash
torchserve --stop
```



