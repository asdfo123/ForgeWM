"""
ForgeWM Data Preparation: encode GameFactory videos into LMDB at 352×640.

IMPORTANT: The training configs expect latent shape (21, 16, 44, 80) which
corresponds to 352×640 pixel resolution. If you encode at a different
resolution (e.g., 480×832 → 60×104 latent), Stage 1+ will fail because
the noise shape won't match the data.

Usage:
    python scripts/prepare_data.py \
        --input_dir /path/to/gamefactory/videos \
        --output_dir ./data/action_lmdb \
        --resolution 352x640 \
        --num_shards 10

Requirements:
    - GameFactory Minecraft dataset (mp4 videos + action labels)
    - Wan2.1 VAE checkpoint at ./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Encode GameFactory data into LMDB")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to GameFactory video directory")
    parser.add_argument("--output_dir", type=str, default="./data/action_lmdb",
                        help="Output LMDB directory")
    parser.add_argument("--resolution", type=str, default="352x640",
                        choices=["352x640", "480x832"],
                        help="Target resolution. Must match config's image_or_video_shape. "
                             "352x640 → latent 44×80 (default, matches all provided configs). "
                             "480x832 → latent 60×104 (requires updating configs).")
    parser.add_argument("--num_shards", type=int, default=10)
    parser.add_argument("--vae_path", type=str,
                        default="./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    parser.add_argument("--num_frames", type=int, default=21,
                        help="Latent frames per clip (default 21 = 81 pixel frames)")
    args = parser.parse_args()

    h, w = [int(x) for x in args.resolution.split("x")]
    latent_h, latent_w = h // 8, w // 8
    print(f"Resolution: {h}×{w} → latent {latent_h}×{latent_w}")
    print(f"Output: {args.output_dir} ({args.num_shards} shards)")
    print(f"Frames per clip: {args.num_frames} latent = {(args.num_frames-1)*4+1} pixel")
    print()

    if latent_h != 44 or latent_w != 80:
        print("⚠️  WARNING: The provided training configs expect latent shape "
              "(21, 16, 44, 80) = 352×640.")
        print("   If you use a different resolution, you MUST update "
              "image_or_video_shape in all configs.")
        print()

    # TODO: Full implementation requires:
    # 1. Load GameFactory videos + action CSVs
    # 2. Resize to target resolution
    # 3. Encode through VAE in chunks of (num_frames-1)*4+1 pixel frames
    # 4. Extract mouse/keyboard action sequences aligned to pixel frames
    # 5. Write to sharded LMDB (latents, mouse_conditions, keyboard_conditions)
    #
    # For now, see the internal prep script for reference:
    #   scripts/prepare_action_lmdb_shard_360p.py (in development repo)
    raise NotImplementedError(
        "Full data preparation pipeline coming soon. "
        "Pre-encoded LMDB will be available on HuggingFace."
    )


if __name__ == "__main__":
    main()
