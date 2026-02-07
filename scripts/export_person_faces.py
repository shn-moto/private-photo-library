"""
Export assigned faces for each person to separate folders.
Creates thumbnails (720p) with cropped face regions.
"""
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageOps
import pillow_heif
import rawpy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from config.settings import settings

# Register HEIF opener
pillow_heif.register_heif_opener()


def load_image_as_pil(file_path: str) -> Optional[Image.Image]:
    """
    Load image from file path and return as PIL Image.
    Supports JPEG, PNG, HEIC, RAW formats.
    Applies EXIF orientation correction.
    """
    try:
        file_lower = file_path.lower()
        
        # RAW formats - rawpy.postprocess() автоматически применяет поворот
        if file_lower.endswith(('.nef', '.cr2', '.arw', '.dng', '.raf', '.orf', '.rw2')):
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess()
                return Image.fromarray(rgb)
        
        # Standard formats (including HEIC through pillow-heif)
        img = Image.open(file_path)
        # CRITICAL: Apply EXIF orientation to match what face detection sees
        img = ImageOps.exif_transpose(img)
        return img
        
    except Exception as e:
        print(f"  ⚠️  Failed to load {file_path}: {e}")
        return None


def resize_to_720p(image: Image.Image) -> Image.Image:
    """Resize image to 720p (height=720, keep aspect ratio)."""
    width, height = image.size
    if height <= 720:
        return image
    
    new_height = 720
    new_width = int(width * (720 / height))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def crop_face_with_margin(image: Image.Image, bbox: tuple, margin: float = 0.3) -> Image.Image:
    """
    Crop face from image with margin around bbox.
    
    Args:
        image: PIL Image
        bbox: (x1, y1, x2, y2) in pixels
        margin: margin around face as fraction of face size (default 30%)
    
    Returns:
        Cropped image with face
    """
    x1, y1, x2, y2 = bbox
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Add margin
    margin_x = face_width * margin
    margin_y = face_height * margin
    
    crop_x1 = max(0, int(x1 - margin_x))
    crop_y1 = max(0, int(y1 - margin_y))
    crop_x2 = min(image.width, int(x2 + margin_x))
    crop_y2 = min(image.height, int(y2 + margin_y))
    
    return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def export_person_faces(
    output_dir: str = "/reports",
    person_id: Optional[int] = None,
    margin: float = 0.3,
    skip_existing: bool = True
):
    """
    Export faces for each person to separate folders.
    
    Args:
        output_dir: Base directory for exports (default: /reports)
        person_id: Export only specific person (optional)
        margin: Margin around face as fraction of face size
        skip_existing: Skip already exported faces
    """
    # Create DB session
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Get persons
        if person_id:
            persons_query = text("""
                SELECT person_id, name
                FROM person
                WHERE person_id = :person_id
            """)
            persons = session.execute(persons_query, {"person_id": person_id}).fetchall()
        else:
            persons_query = text("""
                SELECT person_id, name
                FROM person
                ORDER BY person_id
            """)
            persons = session.execute(persons_query).fetchall()
        
        if not persons:
            print("No persons found")
            return
        
        print(f"Found {len(persons)} person(s)")
        
        # Process each person
        for person_row in persons:
            pid = person_row[0]
            name = person_row[1]
            
            # Create output directory
            person_dir = Path(output_dir) / str(pid) / "faces"
            person_dir.mkdir(parents=True, exist_ok=True)
            
            # Get faces for this person
            faces_query = text("""
                SELECT 
                    f.face_id,
                    f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
                    f.det_score,
                    p.image_id,
                    p.file_path,
                    p.width, p.height
                FROM faces f
                JOIN photo_index p ON f.image_id = p.image_id
                WHERE f.person_id = :person_id
                ORDER BY f.face_id
            """)
            faces = session.execute(faces_query, {"person_id": pid}).fetchall()
            
            if not faces:
                print(f"Person {pid} ({name}): No faces found")
                continue
            
            print(f"\nPerson {pid} ({name}): {len(faces)} face(s)")
            
            # Process each face
            success_count = 0
            skip_count = 0
            error_count = 0
            
            for face_row in tqdm(faces, desc=f"Exporting faces for {name}"):
                face_id = face_row[0]
                bbox = (face_row[1], face_row[2], face_row[3], face_row[4])
                det_score = face_row[5]
                image_id = face_row[6]
                file_path = face_row[7]
                img_width = face_row[8]
                img_height = face_row[9]
                
                # Output filename
                output_file = person_dir / f"face_{face_id}_img_{image_id}_score_{det_score:.2f}.jpg"
                
                if skip_existing and output_file.exists():
                    skip_count += 1
                    continue
                
                try:
                    # Load image
                    image = load_image_as_pil(file_path)
                    
                    if image is None:
                        error_count += 1
                        continue
                    
                    # Check if image dimensions match DB
                    if image.size != (img_width, img_height):
                        # Image was likely rotated by EXIF, need to scale bbox
                        # For simplicity, just use the loaded image dimensions
                        pass
                    
                    # Resize to 720p first (faster processing)
                    orig_height = image.height
                    resized_image = resize_to_720p(image)
                    scale = resized_image.height / orig_height
                    
                    # Scale bbox coordinates
                    scaled_bbox = (
                        bbox[0] * scale,
                        bbox[1] * scale,
                        bbox[2] * scale,
                        bbox[3] * scale
                    )
                    
                    # Crop face with margin
                    face_crop = crop_face_with_margin(resized_image, scaled_bbox, margin)
                    
                    # Convert to RGB if needed (JPEG doesn't support RGBA/transparency)
                    if face_crop.mode in ('RGBA', 'LA', 'P'):
                        face_crop = face_crop.convert('RGB')
                    
                    # Save as JPEG
                    face_crop.save(output_file, "JPEG", quality=90)
                    success_count += 1
                    
                except Exception as e:
                    print(f"  ⚠️  Error processing face {face_id}: {e}")
                    error_count += 1
                    continue
            
            print(f"  ✅ Exported: {success_count}")
            if skip_count > 0:
                print(f"  ⏭️  Skipped: {skip_count}")
            if error_count > 0:
                print(f"  ❌ Errors: {error_count}")
            print(f"  📁 Output: {person_dir}")
    
    finally:
        session.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export person faces to folders")
    parser.add_argument(
        "--person-id",
        type=int,
        help="Export only specific person ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/reports",
        help="Output directory (default: /reports)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
        help="Margin around face (0.0-1.0, default: 0.3 = 30%%)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-export existing faces"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Person Faces Export Tool")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Margin: {args.margin * 100:.0f}%")
    print(f"Skip existing: {not args.no_skip}")
    if args.person_id:
        print(f"Person ID filter: {args.person_id}")
    print("=" * 60)
    
    export_person_faces(
        output_dir=args.output_dir,
        person_id=args.person_id,
        margin=args.margin,
        skip_existing=not args.no_skip
    )
    
    print("\n✅ Export complete!")
