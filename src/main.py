"""
手势识别主程序 - 增强保存功能版
"""
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__))) #添加引用路径

from src.utils import HandDetector
from frame_saver import FrameSaver

def main():
    """主函数，基础识别和保存（尚未添加手部特征计算模块）"""
    print("=== 手势识别系统 v0.3 ===")
    print("快捷键:")
    print("  q - 退出程序")
    print("  s - 保存当前帧")
    print("  a - 切换自动保存模式")
    print("  d - 显示保存信息")
    print("  c - 清空屏幕")
    print()

    # 初始化帧保存器
    saver = FrameSaver("captured_frames")

    # 初始化手部检测器
    detector = HandDetector()

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    print("摄像头已打开，开始检测...")

    frame_count = 0
    auto_save_counter = 0

    while True:
        # 读取一帧
        success, frame = cap.read()

        if not success:
            print("错误：无法读取摄像头画面")
            break

        frame_count += 1
        auto_save_counter += 1

        # 检测手部
        processed_frame = frame.copy()
        processed_frame = detector.detect_hands(processed_frame)

        # === 自动保存逻辑 ===
        if saver.auto_save and auto_save_counter >= saver.auto_save_interval:
            if detector.hand_count > 0:  # 只在检测到手部时自动保存
                success_save, filepath = saver.save_frame(frame, "auto")
                if success_save:
                    print(f"🔄 自动保存: {os.path.basename(filepath)}")
                auto_save_counter = 0

        # === 在图像上绘制信息 ===

        # 1. 手部数量
        hand_text = f"hands: {detector.hand_count}"
        color = (0, 255, 0) if detector.hand_count > 0 else (0, 0, 255)
        cv2.putText(processed_frame, hand_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 2. 帧率
        if frame_count % 10 == 0:
            detector.update_fps()
        cv2.putText(processed_frame, f"FPS: {detector.fps:.1f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 3. 保存状态
        save_status = "auto-saved: ON" if saver.auto_save else "auto-saved: OFF"
        save_color = (0, 255, 0) if saver.auto_save else (0, 0, 255)
        cv2.putText(processed_frame, save_status, 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, save_color, 2)

        # 4. 已保存数量
        cv2.putText(processed_frame, f"saved: {saver.save_count}", 
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # 5. 帮助提示
        help_text = "q:quit s:save a:automatic d:data c:clear"
        cv2.putText(processed_frame, help_text, 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # 显示画面
        cv2.imshow("hang_gesture", processed_frame)

        # === 按键处理 ===
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # 退出
            print("退出程序")
            break

        elif key == ord('s'):  # 手动保存
            success_save, filepath = saver.save_frame(frame, "manual")

            if success_save:
                print(f"✅ 手动保存: {os.path.basename(filepath)}")

                # 显示保存成功提示
                cv2.putText(processed_frame, "save succesfully!", (250, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("saved_frame", processed_frame)
                cv2.waitKey(300)
            else:
                print(f"❌ 保存失败")

        elif key == ord('a'):  # 切换自动保存
            saver.auto_save = not saver.auto_save
            status = "开启" if saver.auto_save else "关闭"
            print(f"自动保存 {status}")

        elif key == ord('d'):  # 显示保存信息
            info = saver.get_save_info()
            print("\n=== 保存信息 ===")
            print(f"总保存数: {info['total_saved']}")
            print(f"保存目录: {info['save_dir']}")
            print(f"自动保存: {'开启' if info['auto_save'] else '关闭'}")
            if info['auto_save']:
                print(f"自动间隔: 每 {info['auto_interval']} 帧")
            print("================")

        elif key == ord('c'):  # 清空屏幕
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=== 手势识别系统 ===")
            print("屏幕已清空")

    # 清理
    cap.release()
    cv2.destroyAllWindows()

    # 最终统计
    print("\n" + "=" * 40)
    print("最终统计:")
    print(f"总处理帧数: {frame_count}")
    print(f"总保存图片: {saver.save_count}")
    print(f"保存目录: {os.path.abspath(saver.base_dir)}")
    print("=" * 40)

if __name__ == "__main__":
    main()