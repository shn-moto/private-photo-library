-- Migration script for face search functionality
-- Run: psql -U dev -d smart_photo_index -f scripts/migrate_faces.sql

-- ============================================
-- Table: person (people/identities)
-- ============================================
CREATE TABLE IF NOT EXISTS person (
    person_id SERIAL PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    description TEXT,
    cover_face_id INTEGER,  -- Best face for avatar (FK added later to avoid circular dependency)
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for name search (case-insensitive)
CREATE INDEX IF NOT EXISTS idx_person_name ON person(name);
CREATE INDEX IF NOT EXISTS idx_person_name_lower ON person(LOWER(name));

-- ============================================
-- Table: faces (detected faces in photos)
-- ============================================
CREATE TABLE IF NOT EXISTS faces (
    face_id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES photo_index(image_id) ON DELETE CASCADE,
    person_id INTEGER REFERENCES person(person_id) ON DELETE SET NULL,

    -- Bounding box (pixel coordinates)
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,

    -- Detection confidence (0.0 - 1.0)
    det_score REAL NOT NULL,

    -- Facial landmarks (5-point or 106-point as JSON array)
    landmarks JSONB,

    -- Estimated attributes from InsightFace
    age INTEGER,
    gender INTEGER,  -- 0 = female, 1 = male

    -- Face embedding vector (InsightFace buffalo_l = 512 dimensions)
    face_embedding vector(512) NOT NULL,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- Foreign key indexes for joins
CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);

-- Index for unassigned faces (for bulk assignment UI)
CREATE INDEX IF NOT EXISTS idx_faces_unassigned ON faces(image_id) WHERE person_id IS NULL;

-- HNSW index for fast face similarity search (cosine distance)
CREATE INDEX IF NOT EXISTS idx_faces_embedding_hnsw
    ON faces USING hnsw (face_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================
-- Add foreign key for person.cover_face_id
-- ============================================
ALTER TABLE person
    DROP CONSTRAINT IF EXISTS fk_person_cover_face;

ALTER TABLE person
    ADD CONSTRAINT fk_person_cover_face
    FOREIGN KEY (cover_face_id) REFERENCES faces(face_id) ON DELETE SET NULL;

-- ============================================
-- Helper function: get face count for person
-- ============================================
CREATE OR REPLACE FUNCTION get_person_face_count(p_person_id INTEGER)
RETURNS INTEGER AS $$
BEGIN
    RETURN (SELECT COUNT(*) FROM faces WHERE person_id = p_person_id);
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- View: persons with face counts
-- ============================================
CREATE OR REPLACE VIEW person_with_stats AS
SELECT
    p.person_id,
    p.name,
    p.description,
    p.cover_face_id,
    p.created_at,
    p.updated_at,
    COUNT(f.face_id) AS face_count,
    COUNT(DISTINCT f.image_id) AS photo_count
FROM person p
LEFT JOIN faces f ON f.person_id = p.person_id
GROUP BY p.person_id, p.name, p.description, p.cover_face_id, p.created_at, p.updated_at;

-- ============================================
-- Statistics
-- ============================================
-- Check migration success:
-- SELECT 'person' AS table_name, COUNT(*) AS row_count FROM person
-- UNION ALL
-- SELECT 'faces', COUNT(*) FROM faces;
