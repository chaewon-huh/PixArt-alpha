import os
import json
import shutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

def find_actual_filename(json_filename, source_dir):
    """
    JSON의 파일명을 기반으로 실제 존재하는 파일명을 찾습니다.
    여러 패턴을 시도해봅니다.
    """
    # 시도할 패턴들
    patterns_to_try = [
        json_filename,  # JSON에 있는 그대로
        json_filename.replace("_refined_512.png", "_512.png"),  # _refined_512.png -> _512.png
        json_filename.replace("_refined_512.png", ".png"),  # _refined_512.png -> .png
        json_filename.replace("_refined_512", "_512"),  # _refined_512 -> _512
        json_filename.replace("_refined", ""),  # _refined 제거
        json_filename.replace("refined_", ""),  # refined_ 제거
    ]
    
    # 각 패턴을 순서대로 시도
    for pattern in patterns_to_try:
        full_path = os.path.join(source_dir, pattern)
        if os.path.exists(full_path):
            return pattern
    
    # 모든 패턴이 실패하면 None 반환
    return None

def rename_and_copy_image(args_tuple):
    json_original_filename, new_filename, source_dir, target_dir = args_tuple
    
    # 실제 파일명 찾기
    actual_filename = find_actual_filename(json_original_filename, source_dir)
    
    if actual_filename is None:
        return False, f"Could not find file for '{json_original_filename}' in {source_dir}. Tried multiple patterns."
    
    source_path = os.path.join(source_dir, actual_filename)
    target_path = os.path.join(target_dir, new_filename)
    
    try:
        # Ensure the target subdirectory exists if new_filename includes subdirectories
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(source_path, target_path)
        return True, f"Successfully copied {actual_filename} -> {new_filename}"
    except Exception as e:
        return False, f"Error copying {actual_filename} to {new_filename}: {e}"

def main(args):
    # Load the name mapping from the JSON file
    try:
        with open(args.name_map_json, 'r') as f:
            name_map_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: Name map JSON file not found at {args.name_map_json}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.name_map_json}")
        return

    # Prepare arguments for multiprocessing
    tasks = []
    for item in name_map_list:
        original_filename = item.get('original_filename')  # JSON의 실제 키
        new_filename = item.get('new_filename')  # JSON의 실제 키

        if not original_filename or not new_filename:
            print(f"Warning: Skipping invalid item in JSON (original_filename or new_filename missing): {item}")
            continue
        
        tasks.append((original_filename, new_filename, args.source_dir, args.target_dir))

    # Create target directory if it doesn't exist
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
        print(f"Created target directory: {args.target_dir}")

    # Use multiprocessing to rename and copy files
    num_processes = min(args.num_processes, cpu_count())
    print(f"Starting renaming and copying with {num_processes} processes...")
    print(f"Processing {len(tasks)} files...")

    success_count = 0
    error_count = 0
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(rename_and_copy_image, tasks), total=len(tasks)))
    
    for success, message in results:
        if success:
            success_count += 1
            if args.verbose:
                print(message)  # 성공 메시지도 verbose일 때만 출력
        else:
            error_count += 1
            print(message)  # 에러는 항상 출력

    print(f"\nRenaming and copying complete.")
    print(f"Successfully processed: {success_count} files.")
    print(f"Errors encountered: {error_count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename and copy images based on a JSON mapping using multiprocessing.")
    parser.add_argument("--name_map_json", type=str, default="./name_debug.json",
                        help="Path to the JSON file containing original_filename and new_filename mappings.")
    parser.add_argument("--source_dir", type=str, default="/workspace/PixArt-alpha/data_raw/FinalInputData512",
                        help="Directory containing the original images.")
    parser.add_argument("--target_dir", type=str, default="/workspace/PixArt-alpha/data_raw/FinalInputData512_renamed",
                        help="Directory to save the renamed images.")
    parser.add_argument("--num_processes", type=int, default=cpu_count(),
                        help="Number of worker processes to use. Defaults to CPU count.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print verbose output including successful operations.")
    
    args = parser.parse_args()
    main(args) 