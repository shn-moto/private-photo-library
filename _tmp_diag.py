import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("192.168.1.107", username="photolib", password="photolib")

# Check a few photos: compare DB dims vs expected from EXIF
# Get some photos with EXIF orientation containing 90 or 270
cmd = """docker exec smart_photo_db psql -U dev -d smart_photo_index -t -c "
SELECT image_id, width, height, file_format,
       exif_data->>'Orientation' as orientation
FROM photo_index
WHERE exif_data->>'Orientation' LIKE '%90%'
   OR exif_data->>'Orientation' LIKE '%270%'
LIMIT 10;
" """
print("=== Photos with 90/270 EXIF orientation ===")
_, stdout, stderr = ssh.exec_command(cmd, timeout=30)
print(stdout.read().decode())
err = stderr.read().decode()
if err: print("ERR:", err)

# Also check if fix-dimensions was in logs
cmd2 = """docker logs smart_photo_api 2>&1 | grep -i "fix.*dimension" | tail -5"""
print("\n=== API logs mentioning fix-dimensions ===")
_, stdout, stderr = ssh.exec_command(cmd2, timeout=15)
print(stdout.read().decode())
err = stderr.read().decode()
if err: print("ERR:", err)

# Check total photos where width > height (landscape) but orientation says portrait
cmd3 = """docker exec smart_photo_db psql -U dev -d smart_photo_index -t -c "
SELECT COUNT(*) as total,
       COUNT(*) FILTER (WHERE width > height AND (exif_data->>'Orientation' LIKE '%90%' OR exif_data->>'Orientation' LIKE '%270%')) as rotated_landscape
FROM photo_index
WHERE width IS NOT NULL;
" """
print("\n=== Stats: rotated photos stored as landscape (potential double-swap) ===")
_, stdout, stderr = ssh.exec_command(cmd3, timeout=15)
print(stdout.read().decode())
err = stderr.read().decode()
if err: print("ERR:", err)

ssh.close()
print("\nDone!")
