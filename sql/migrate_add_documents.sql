-- Document photo tool: RBAC section + function.
-- Run: psql -U dev -d smart_photo_index -f sql/migrate_add_documents.sql

INSERT INTO api_section (section_code, section_name, description, is_public, is_admin_only, sort_order)
VALUES ('documents', 'Фото на документы', 'Подготовка фото на документы через AI-ассистента', FALSE, FALSE, 27)
ON CONFLICT (section_code) DO NOTHING;

INSERT INTO api_function (function_code, section_code, function_name, description, sort_order)
VALUES ('documents.create', 'documents', 'Готовить фото на документы',
        'Кроп по стандарту, замена фона, скачивание и сохранение результата (/documents/*)', 1)
ON CONFLICT (function_code) DO NOTHING;

-- Grant to every existing user (new users get all assignable functions on creation)
INSERT INTO user_function_permission (user_id, function_code)
SELECT au.user_id, 'documents.create'
FROM app_user au
WHERE NOT EXISTS (
    SELECT 1 FROM user_function_permission ufp
    WHERE ufp.user_id = au.user_id AND ufp.function_code = 'documents.create'
);
