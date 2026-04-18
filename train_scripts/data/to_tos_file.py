import os

for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

import tos
from tos import DataTransferType
import argparse
import time

# Credentials via env — do NOT hardcode. Export before running:
#   export VOLC_TOS_AK=...  VOLC_TOS_SK=...
ak = os.environ.get("VOLC_TOS_AK") or os.environ.get("VOLCENGINE_ACCESS_KEY", "")
sk = os.environ.get("VOLC_TOS_SK") or os.environ.get("VOLCENGINE_SECRET_KEY", "")
if not ak or not sk:
    raise SystemExit("VOLC_TOS_AK / VOLC_TOS_SK not set in env")
endpoint = os.environ.get("VOLC_TOS_ENDPOINT", "tos-cn-shanghai.volces.com")
region = os.environ.get("VOLC_TOS_REGION", "cn-shanghai")
bucket_name = os.environ.get("VOLC_TOS_BUCKET", "transfer-shanghai")

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='本地文件路径')
parser.add_argument('--object_key', type=str, required=True, help='TOS 对象键（含路径）')
parser.add_argument('--task_num', type=int, default=16, help='分片并发数')
parser.add_argument('--part_size_mb', type=int, default=64, help='分片大小 MB')
args = parser.parse_args()

client = tos.TosClientV2(ak, sk, endpoint, region)

last_print = [0.0]
def percentage(consumed_bytes, total_bytes, rw_once_bytes, type):
    now = time.time()
    if total_bytes and (now - last_print[0] > 2 or consumed_bytes == total_bytes):
        rate = 100 * consumed_bytes / total_bytes
        print(f"progress: {rate:.2f}% ({consumed_bytes}/{total_bytes})", flush=True)
        last_print[0] = now

try:
    client.head_object(bucket_name, args.object_key)
    print(f"{args.object_key} 已存在，跳过")
except Exception:
    print(f"开始上传: {args.file} -> {args.object_key}", flush=True)
    t0 = time.time()
    client.upload_file(
        bucket_name, args.object_key, args.file,
        task_num=args.task_num,
        part_size=1024 * 1024 * args.part_size_mb,
        data_transfer_listener=percentage,
    )
    dt = time.time() - t0
    size = os.path.getsize(args.file)
    print(f"完成: 耗时 {dt:.1f}s, 平均 {size/1024/1024/dt:.2f} MB/s")
