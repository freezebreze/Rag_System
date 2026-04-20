# -*- coding: utf-8 -*-
"""
对象存储服务
负责文件上传和临时 URL 生成
"""

import logging
import datetime
from typing import Optional
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from app.core.config import settings

logger = logging.getLogger(__name__)


class OSSService:
    """对象存储服务，支持 Aliyun OSS 和 Cloudflare R2。"""

    def __init__(self):
        self.provider = settings.object_storage_provider
        if self.provider == "r2":
            self._init_r2_client()
            return

        self._init_aliyun_client()

    def _init_aliyun_client(self) -> None:
        import alibabacloud_oss_v2 as oss

        self.oss = oss
        credentials_provider = oss.credentials.StaticCredentialsProvider(
            access_key_id=settings.oss_access_key_id,
            access_key_secret=settings.oss_access_key_secret,
        )
        cfg = oss.config.load_default()
        cfg.credentials_provider = credentials_provider
        cfg.region = settings.oss_region
        cfg.endpoint = settings.oss_endpoint
        cfg.insecure_skip_verify = True
        self.client = oss.Client(cfg)
        self.bucket = settings.oss_bucket
        logger.info(f"OSS 服务初始化: bucket={self.bucket}, region={settings.oss_region}")

    def _init_r2_client(self) -> None:
        import boto3
        from botocore.config import Config

        self.client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name=settings.r2_region,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
        self.bucket = settings.r2_bucket
        logger.info(f"R2 服务初始化: bucket={self.bucket}, endpoint={settings.r2_endpoint}")

    def upload_bytes(self, object_key: str, file_content: bytes) -> str:
        """
        直接用完整 object_key 上传（document_service 专用）

        Args:
            object_key: 完整 OSS 路径，如 kb/my_kb/file.pdf
            file_content: 文件二进制内容

        Returns:
            object_key
        """
        try:
            if self.provider == "r2":
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=object_key,
                    Body=file_content,
                )
                logger.info(f"R2 上传成功: {object_key}")
                return object_key

            result = self.client.put_object(
                self.oss.PutObjectRequest(
                    bucket=self.bucket,
                    key=object_key,
                    body=file_content,
                )
            )
            logger.info(f"OSS 上传成功: {object_key}, status={result.status_code}")
            return object_key
        except Exception as e:
            logger.error(f"OSS 上传失败: {object_key}, error={e}")
            raise Exception(f"OSS 上传失败: {e}")

    def upload_file(self, category_name: str, file_name: str, file_content: bytes) -> str:
        """
        上传文件到 OSS

        Args:
            category_name: 类目名称，作为 OSS 目录
            file_name: 文件名
            file_content: 文件二进制内容

        Returns:
            object_key: OSS 对象路径，格式为 类目名/文件名
        """
        object_key = f"{category_name}/{file_name}"
        try:
            if self.provider == "r2":
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=object_key,
                    Body=file_content,
                )
                logger.info(f"R2 上传成功: {object_key}")
                return object_key

            result = self.client.put_object(
                self.oss.PutObjectRequest(
                    bucket=self.bucket,
                    key=object_key,
                    body=file_content,
                )
            )
            logger.info(f"OSS 上传成功: {object_key}, status={result.status_code}")
            return object_key
        except Exception as e:
            logger.error(f"OSS 上传失败: {object_key}, error={e}")
            raise Exception(f"OSS 上传失败: {e}")

    def get_object_bytes(self, object_key: str) -> bytes:
        """用 SDK 直接下载 OSS 对象，返回字节内容（后端专用，不需要预签名）"""
        try:
            if self.provider == "r2":
                result = self.client.get_object(Bucket=self.bucket, Key=object_key)
                body = result["Body"]
                try:
                    data = body.read()
                finally:
                    body.close()
                logger.info(f"R2 下载成功: {object_key}, size={len(data)}")
                return data

            result = self.client.get_object(
                self.oss.GetObjectRequest(bucket=self.bucket, key=object_key)
            )
            with result.body as body:
                data = body.read()
            logger.info(f"OSS 下载成功: {object_key}, size={len(data)}")
            return data
        except Exception as e:
            logger.error(f"OSS 下载失败: {object_key}, error={e}")
            raise Exception(f"OSS 下载失败: {e}")

    def delete_objects(self, object_keys: list) -> int:
        """批量删除 OSS 对象，返回实际删除数量。失败不抛异常，只记录日志。"""
        if not object_keys:
            return 0
        try:
            if self.provider == "r2":
                deleted_count = 0
                for i in range(0, len(object_keys), 1000):
                    batch = object_keys[i: i + 1000]
                    result = self.client.delete_objects(
                        Bucket=self.bucket,
                        Delete={
                            "Objects": [{"Key": key} for key in batch],
                            "Quiet": False,
                        },
                    )
                    deleted_count += len(result.get("Deleted", []))
                logger.info(f"R2 批量删除: 请求 {len(object_keys)} 个, 成功 {deleted_count} 个")
                return deleted_count

            objects = [self.oss.DeleteObject(key=k) for k in object_keys]
            result = self.client.delete_multiple_objects(
                self.oss.DeleteMultipleObjectsRequest(bucket=self.bucket, objects=objects)
            )
            # alibabacloud_oss_v2 默认非静默模式：deleted_objects 包含所有成功删除的对象
            # 失败的对象不在此列表中（OSS 对不存在的对象也视为删除成功）
            deleted_count = len(result.deleted_objects) if result.deleted_objects else 0
            failed = len(object_keys) - deleted_count
            logger.info(f"OSS 批量删除: 请求 {len(object_keys)} 个, 成功 {deleted_count} 个, 失败 {failed} 个")
            return deleted_count
        except Exception as e:
            logger.error(f"OSS 批量删除失败: {e}")
            return 0

    def get_presigned_url(self, object_key: str, expires: int = 3600) -> str:
        """
        生成临时签名 URL

        Args:
            object_key: OSS 对象路径
            expires: 有效期（秒），默认 3600

        Returns:
            临时访问 URL
        """
        try:
            if self.provider == "r2":
                url = self.client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": object_key},
                    ExpiresIn=expires,
                )
                logger.info(f"生成 R2 临时 URL: {object_key}, expires={expires}s")
                return url

            result = self.client.presign(
                self.oss.GetObjectRequest(
                    bucket=self.bucket,
                    key=object_key,
                ),
                expires=datetime.timedelta(seconds=expires),
            )
            logger.info(f"生成临时 URL: {object_key}, expires={expires}s")
            return result.url
        except Exception as e:
            logger.error(f"生成临时 URL 失败: {object_key}, error={e}")
            raise Exception(f"生成临时 URL 失败: {object_key}, error={e}")

    def get_presigned_url_by_category(
        self, category_name: str, file_name: str, expires: int = 3600
    ) -> str:
        """通过类目名和文件名生成临时 URL"""
        object_key = f"{category_name}/{file_name}"
        return self.get_presigned_url(object_key, expires)


_instance: Optional[OSSService] = None


def get_oss_service() -> OSSService:
    global _instance
    if _instance is None:
        _instance = OSSService()
    return _instance
