import cv2
import numpy as np
import os
from glob import glob

def create_scrolling_video_wall(video_paths, output_path, fps=30, duration=30, resolution=(1920, 1080)):
    """
    创建一个滚动视频墙，将多个512×512视频拼接成一个高分辨率视频
    
    参数:
    video_paths: 输入视频路径列表
    output_path: 输出视频路径
    fps: 输出视频帧率
    duration: 视频总时长(秒)
    resolution: 输出视频分辨率，默认1080p
    """
    if not video_paths:
        print("没有提供视频文件")
        return False
    
    width, height = resolution
    video_size = 512  # 输入视频的尺寸（假设是正方形）
    
    # 计算视频墙布局
    videos_per_row = (width // video_size) + 2  # 多放两个用于滚动
    rows = 2  # 分两行显示
    
    # 计算垂直居中对齐的偏移量
    row_height = video_size
    total_video_height = rows * row_height
    vertical_offset = (height - total_video_height) // 2  # 垂直居中
    
    total_frames = int(fps * duration)
    scroll_width = len(video_paths) * video_size  # 所有视频拼接的总宽度
    
    # 初始化输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    if not out.isOpened():
        print("无法创建输出视频文件")
        return False
    
    # 从每个视频中提取第一帧，用于创建视频墙
    video_frames = []
    caps = []
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            continue
        
        caps.append(cap)
        ret, frame = cap.read()
        if ret:
            # 调整帧大小
            frame = cv2.resize(frame, (video_size, video_size))
            video_frames.append(frame)
        else:
            print(f"无法读取视频帧: {video_path}")
    
    # 关闭读取的视频
    for cap in caps:
        cap.release()
    
    # 重新打开视频准备提取所有帧
    all_video_frames = []
    frame_counts = []
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (video_size, video_size))
            frames.append(frame)
        
        cap.release()
        
        if frames:
            all_video_frames.append(frames)
            frame_counts.append(len(frames))
        else:
            print(f"视频没有帧: {video_path}")
    
    # 找到最短的视频长度，确保循环
    min_frames = min(frame_counts) if frame_counts else 0
    if min_frames == 0:
        print("没有有效的视频帧")
        return False
    
    # 为每个视频规范帧数量
    normalized_videos = []
    for frames in all_video_frames:
        # 确保每个视频都有相同的帧数，循环使用帧
        normalized = []
        for i in range(min_frames):
            normalized.append(frames[i % len(frames)])
        normalized_videos.append(normalized)
    
    # 创建滚动效果
    for frame_idx in range(total_frames):
        # 计算滚动偏移量
        scroll_offset = (frame_idx / total_frames) * scroll_width
        scroll_offset %= scroll_width  # 循环滚动
        
        # 创建一个白色画布 (而不是黑色)
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 填充两行视频
        for row in range(rows):
            row_y = row * video_size + vertical_offset  # 应用垂直居中偏移
            
            # 如果第二行，反向滚动
            if row == 1:
                current_offset = scroll_width - scroll_offset
            else:
                current_offset = scroll_offset
            
            # 绘制视频墙的当前可见部分
            x_start = -int(current_offset % scroll_width)
            
            # 循环放置视频，直到填满整个宽度
            while x_start < width:
                # 找出当前位置应该显示哪个视频
                video_index = int((x_start + current_offset) / video_size) % len(normalized_videos)
                
                # 获取当前视频的当前帧
                if row == 1:
                    # 第二行视频索引反向排列
                    reversed_video_index = len(normalized_videos) - 1 - video_index
                    # 第二行视频帧反向播放
                    current_frame = normalized_videos[reversed_video_index][(min_frames - 1) - (frame_idx % min_frames)]
                else:
                    current_frame = normalized_videos[video_index][frame_idx % min_frames]
                
                # 计算视频在画布上的位置
                left = x_start
                right = left + video_size
                top = row_y
                bottom = top + video_size
                
                # 如果视频部分可见，绘制它
                if right > 0 and left < width:
                    # 计算需要绘制的视频部分
                    canvas_left = max(0, left)
                    canvas_right = min(width, right)
                    canvas_top = top
                    canvas_bottom = min(height, bottom)
                    
                    # 计算视频帧中对应的部分
                    frame_left = max(0, canvas_left - left)
                    frame_right = video_size - max(0, right - width)
                    frame_top = max(0, canvas_top - top)
                    frame_bottom = video_size - max(0, bottom - height)
                    
                    # 绘制视频帧到画布上
                    canvas[canvas_top:canvas_bottom, canvas_left:canvas_right] = current_frame[frame_top:frame_bottom, frame_left:frame_right]
                
                # 移动到下一个位置
                x_start += video_size
        
        # 写入当前帧到输出视频
        out.write(canvas)
    
    # 释放资源
    out.release()
    print(f"视频墙已创建: {output_path}")
    return True

# 获取要处理的视频文件列表
def get_video_files_list():
    """返回要处理的视频文件列表，包含完整路径"""
    base_folder = "outputs_to_release_2/selected_videos_4_gif"
    
    video_files = [
        "Medusa_wearing_a_sunglass_and_shopping_with_a_snake_around_her_neck.mp4",
        "Dungeons_and_Dragons_Bugbear_Merchant_fat_many_ring_piercings_creepy_smile.mp4",
        "Dungeons_and_Dragons_anime_death_knight_riding_a_fire_horse.mp4",
        "A_monkey_with_a_white_suit_and_ski_goggles_laughing_surfing_in_the_sea_hyper-realistic_award-winning_animation_style.mp4",
        "Elf_knight_order_in_fantasy_setting.mp4",
        "Cleopatra_wearing_VR_glasses_and_earphones_typing_on_a_laptop_white_dress_animation_style_cyberpunk.mp4",
        "Dungeons_and_Dragons_by_Arthur_Sarnoff_and_Dean_Ellis.mp4",
        "The_orc_wearing_a_gray_hat_is_reading_a_book.mp4",
        "Dante_from_Devil_May_Cry_dressed_in_tactical_gear_realistic_full_body_pose.mp4",
        "An_astronaut_riding_a_sea_turtle_hyper-realistic_award-winning_advertisement_4K_HD.mp4",
        "A_dog_is_jumping_to_catch_the_flower.mp4",
        "A_happy_moment_as_a_bearded_man_with_a_bald_head_finds_a_key.mp4",
        "Hyper-realistic_Japanese_dragon_full_8K.mp4",
        "A_hobbit_riding_a_train_in_a_police_station_digital_art_highly_detailed.mp4",
        "A_fantasy_version_of_Captain_America_wearing_white_armor.mp4",
        "A_hamster_wearing_a_top_hat_and_suit_imagining_kicking_a_football_award-winning_realistic_painting.mp4",
        "A_dark_tyranids_mecha_gundam_style.mp4",
        "The_policewoman_with_a_gas_mask.mp4",
        "Arnold_Schwarzenegger_shirt_suit_shirtless_muscle.mp4",
        "Cerebro_from_X-Men_as_an_unexplored_wilderness_comforting_colors_peaceful_inspiring.mp4",
        "A_hobbit_with_red_hair_holding_a_compass_in_a_plain_portrait_award-winning.mp4",
        "Dragon_tiger_Victorian_art_style.mp4",
        "A_hobbit_with_silver_hair_planting_raspberries_in_a_cafeteria_graffiti_art_highly_detailed.mp4",
        "Female_beauty_by_the_standards_of_5th_century_Europe.mp4",
        "A_bearded_professional_bald_poker_player_holding_two_cards_by_Ron_English.mp4",
        "Godzilla_roaring_to_the_sky.mp4",
        "Beautiful_Elsa_princess_eating_ice_cream_in_a_snowy_wonderland_fantasy_style_hyper-realistic.mp4",
        "Fat_Australian_woman_penguin_island_grotesque.mp4",
        "A_black_Dragonborn_Bard_that_plays_an_ocarina_in_a_fantasy_setting.mp4",
        "Jared_Leto's_Joker_in_the_style_of_The_Batman_Animated_Series_episode_screencapture.mp4",
        "Dinosaur_in_New_York_by_Jean_Dubuffet.mp4",
        "Ghost_on_skateboard_cartoon_style.mp4",
        "Dungeons_and_Dragons_College_of_Whispers_Bard_Changeling_male_holding_a_Venetian_mask.mp4",
        "Grandma_is_kissing_a_baby_detailed_(Renaissance_style).mp4",
        "Henry_Cavill_as_a_gladiator_Maximus_fighting.mp4",
        "Donald_Trump_mixed_up_with_Superman's_suit_animation_avatar_style_extremely_realistic.mp4",
        "A_goblin_driving_a_snowmobile_in_a_cave_movie_poster_highly_detailed.mp4",
        "Death_god_wearing_a_cloak_playing_video_games.mp4"
    ]
    
    # 构建完整路径列表
    full_paths = [os.path.join(base_folder, file_name) for file_name in video_files]
    
    return full_paths

# 主函数
if __name__ == "__main__":
    import os
    
    # 获取视频文件列表
    video_paths = get_video_files_list()
    
    # 输出视频文件
    output_path = "scrolling_video_wall.mp4"
    
    # 创建视频墙
    create_scrolling_video_wall(
        video_paths=video_paths,
        output_path=output_path,
        fps=30,
        duration=20,  # 20秒长的视频
        resolution=(1920, 1080)  # 1080p分辨率
    )
    
    print(f"处理完成! 输出文件: {output_path}")