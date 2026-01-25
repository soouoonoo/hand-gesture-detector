import cv2
import os
import time

class FrameSaver:
    """帧保存管理器"""
    def __init__(self, base_dir="saved_frames"):
        self.base_dir = base_dir
        self.save_count = 0
        self.auto_save = False  # 自动保存模式
        self.auto_save_interval = 10  # 自动保存间隔（帧数）

        # 创建目录
        self.create_directories()

    def create_directories(self):
        """创建保存目录结构"""
        # 主目录
        os.makedirs(self.base_dir, exist_ok=True)

        # 子目录：按日期分类
        date_str = time.strftime("%Y%m%d")
        self.today_dir = os.path.join(self.base_dir, date_str)
        os.makedirs(self.today_dir, exist_ok=True)

        print(f"保存目录: {os.path.abspath(self.today_dir)}")

    def save_frame(self, frame, prefix="hand"):
        """保存一帧图像"""
        self.save_count += 1

        # 生成文件名
        timestamp = time.strftime("%H%M%S")
        filename = f"{prefix}_{timestamp}_{self.save_count:04d}.jpg"
        filepath = os.path.join(self.today_dir, filename)

        # 保存图像
        success = cv2.imwrite(filepath, frame)

        if success:
            return True, filepath
        else:
            return False, filepath

    def get_save_info(self):
        """获取保存信息"""
        return {
            "total_saved": self.save_count,
            "save_dir": self.today_dir,
            "auto_save": self.auto_save,
            "auto_interval": self.auto_save_interval
        }
