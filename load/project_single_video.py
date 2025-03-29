import cv2
import numpy as np
import os
from glob import glob
import subprocess

def create_video_from_video_quarters(input_video_path, output_path, fps=None, transition_frames=15):
    """
    从输入视频创建新视频，将视频分为四等份，使用第一等份和第二等份，
    并在切换时添加平滑过渡效果
    
    参数:
    input_video_path: 输入视频的文件路径
    output_path: 输出视频的文件路径
    fps: 输出视频的帧率，如果为None则使用输入视频的帧率
    transition_frames: 过渡时使用的帧数
    """
    # 检查输入视频是否存在
    if not os.path.isfile(input_video_path):
        print(f"输入视频文件不存在: {input_video_path}")
        return False
    
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_video_path}")
        return False
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 如果未指定fps，使用原始视频的fps
    if fps is None:
        fps = original_fps
    
    # 计算1/4宽度
    quarter_width = width // 4
    
    # 读取所有帧并分割为四等份，取第一份和第二份
    first_quarter_frames = []  # 0到1/4宽度
    second_quarter_frames = []  # 1/4到2/4宽度
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 分割图像为四等份，并获取第一份和第二份
        first_quarter = frame[:, 0:quarter_width]
        second_quarter = frame[:, quarter_width:quarter_width*2]
        
        first_quarter_frames.append(first_quarter)
        second_quarter_frames.append(second_quarter)
    
    cap.release()
    
    # 确保成功读取了帧
    if not first_quarter_frames:
        print(f"无法从视频中读取帧: {input_video_path}")
        return False
    
    # 计算区间边界
    total_frames = len(first_quarter_frames)
    time_quarter = total_frames // 4  # 时间上的四分之一
    
    # 创建输出帧列表，包括过渡效果
    output_frames = []
    
    # 前1/4区域 - 使用第一份
    for i in range(0, time_quarter - transition_frames):
        output_frames.append(first_quarter_frames[i])
    
    # 第一个过渡区域：从第一份到第二份
    for i in range(transition_frames * 2):
        if time_quarter - transition_frames + i < total_frames and i < transition_frames * 2:
            idx = time_quarter - transition_frames + i
            # 计算混合权重
            alpha = i / (transition_frames * 2)
            # 混合两个四分之一区域
            blended = cv2.addWeighted(first_quarter_frames[idx], 1 - alpha, second_quarter_frames[idx], alpha, 0)
            output_frames.append(blended)
    
    # 中间区域 - 使用第二份
    for i in range(time_quarter + transition_frames, 3 * time_quarter - transition_frames):
        output_frames.append(second_quarter_frames[i])
    
    # 第二个过渡区域：从第二份到第一份
    for i in range(transition_frames * 2):
        if 3 * time_quarter - transition_frames + i < total_frames and i < transition_frames * 2:
            idx = 3 * time_quarter - transition_frames + i
            # 计算混合权重
            alpha = i / (transition_frames * 2)
            # 混合第二份到第一份
            blended = cv2.addWeighted(second_quarter_frames[idx], 1 - alpha, first_quarter_frames[idx], alpha, 0)
            output_frames.append(blended)
    
    # 最后1/4区域 - 使用第一份
    for i in range(3 * time_quarter + transition_frames, total_frames):
        if i < total_frames:
            output_frames.append(first_quarter_frames[i])
    
    # 尝试多种MP4兼容的编解码器
    codecs_to_try = ['mp4v', 'avc1', 'X264']
    temp_output = output_path
    success = False
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            print(f"尝试使用编解码器: {codec}")
            
            # 创建输出视频
            video_writer = cv2.VideoWriter(temp_output, fourcc, fps, (quarter_width, height))
            
            if not video_writer.isOpened():
                print(f"编解码器 {codec} 初始化失败，尝试下一个")
                continue
            
            # 写入所有帧到视频
            for frame in output_frames:
                video_writer.write(frame)
            
            video_writer.release()
            success = True
            print(f"视频已使用 {codec} 编解码器保存到 {temp_output}")
            break
        except Exception as e:
            print(f"使用编解码器 {codec} 时出错: {e}")
    
    # 如果所有MP4编解码器都失败，则回退到使用AVI格式
    if not success:
        print("所有MP4编解码器都失败。回退到使用MJPG编解码器和AVI格式。")
        
        # 使用MJPG编解码器和AVI格式
        avi_output = os.path.splitext(output_path)[0] + "_temp.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(avi_output, fourcc, fps, (quarter_width, height))
        
        if not video_writer.isOpened():
            print("无法创建输出视频文件，请检查编解码器支持和输出路径")
            return False
        
        # 写入所有帧到视频
        for frame in output_frames:
            video_writer.write(frame)
        
        video_writer.release()
        
        # 提示用户如何将AVI转换为MP4
        print(f"已创建临时AVI文件: {avi_output}")
        print(f"您可以使用FFmpeg等工具将其转换为MP4格式: ")
        print(f"ffmpeg -i {avi_output} -c:v libx264 {output_path}")
        
    return success

def process_folder(input_folder, output_folder, video_extensions=None, transition_frames=15):
    """
    处理文件夹中的所有视频文件
    
    参数:
    input_folder: 输入视频文件夹
    output_folder: 输出视频文件夹
    video_extensions: 视频文件扩展名列表，如果为None则使用默认列表
    transition_frames: 过渡帧数
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有视频文件
    video_files = []
    for ext in video_extensions:
        pattern = os.path.join(input_folder, f'*{ext}')
        video_files.extend(glob(pattern))
    
    if not video_files:
        print(f"在 {input_folder} 中没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件需要处理")
    
    # 处理每个视频文件
    success_count = 0
    failure_count = 0
    
    for video_file in video_files:
        # 获取文件名和扩展名
        base_name = os.path.basename(video_file)
        output_path = os.path.join(output_folder, base_name)
        
        print(f"\n处理视频: {base_name}")
        
        # 处理视频
        result = create_video_from_video_quarters(video_file, output_path, transition_frames=transition_frames)
        
        if result:
            success_count += 1
        else:
            failure_count += 1
    
    print(f"\n处理完成! 成功: {success_count}, 失败: {failure_count}")

# 使用示例
if __name__ == "__main__":
    # 单个文件处理
    # input_video = "outputs_to_release_2/selected_views/20_year_old_Serbian_with_brown_curly_mullet_in_Naruto_art_form.mp4"
    # output_video = "output_processed_video.mp4"
    # create_video_from_video_quarters(input_video, output_video, transition_frames=15)
    
    # 文件夹批量处理
    input_folder = "outputs_to_release_2/selected_views"  # 包含原始视频的文件夹
    output_folder = "outputs_to_release_2/processed_videos"  # 输出处理后视频的文件夹
    process_folder(input_folder, output_folder, transition_frames=15)
