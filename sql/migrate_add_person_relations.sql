-- Migration: add birth_date to person, create person_relation table
-- Run: psql -U dev -d smart_photo_index -f sql/migrate_add_person_relations.sql

-- 1. Add birth_date to person table
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'person' AND column_name = 'birth_date'
    ) THEN
        ALTER TABLE person ADD COLUMN birth_date DATE;
    END IF;
END $$;

-- 2. Create person_relation table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'person_relation') THEN
        CREATE TABLE person_relation (
            relation_id SERIAL PRIMARY KEY,
            person_id_from INTEGER NOT NULL REFERENCES person(person_id) ON DELETE CASCADE,
            person_id_to INTEGER NOT NULL REFERENCES person(person_id) ON DELETE CASCADE,
            relation_type VARCHAR(32) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            CONSTRAINT chk_no_self_relation CHECK (person_id_from <> person_id_to),
            CONSTRAINT uq_person_relation UNIQUE (person_id_from, person_id_to, relation_type)
        );
        CREATE INDEX idx_person_relation_from ON person_relation(person_id_from);
        CREATE INDEX idx_person_relation_to ON person_relation(person_id_to);
        -- relation_type values: 'parent' (from is parent of to), 'spouse' (symmetric)
    END IF;
END $$;
