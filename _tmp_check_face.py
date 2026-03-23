import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.1.107', username='photolib', password='photolib')

# Check DB dims + PIL dims for several faces (old OK vs new broken)
# Need file paths from DB
cmd = '''docker exec smart_photo_db psql -U dev -d smart_photo_index -t -A -c "
SELECT f.face_id, f.image_id, p.width, p.height, 
       p.exif_data->>'Orientation' as orient,
       p.file_path
FROM faces f JOIN photo_index p ON f.image_id = p.image_id
WHERE f.face_id IN (40493, 40505, 40538, 50195, 40801, 40821)
ORDER BY f.face_id"
'''
stdin, stdout, stderr = ssh.exec_command(cmd)
rows = stdout.read().decode().strip().split('\n')
print("=== DB data ===")
for r in rows:
    parts = r.split('|')
    print(f"face={parts[0]:>6} img={parts[1]:>6} DB={parts[2]:>5}x{parts[3]:<5} orient={parts[4]:>25} path=...{parts[5][-40:]}")

# Now test PIL dims on server for each
paths = [r.split('|')[5] for r in rows if r.strip()]
face_ids = [r.split('|')[0] for r in rows if r.strip()]
db_dims = [(r.split('|')[2], r.split('|')[3]) for r in rows if r.strip()]

script_lines = [
    'from PIL import Image, ImageOps',
    'from pillow_heif import register_heif_opener',
    'register_heif_opener()',
    '',
]
for i, (fp, fid, (dbw, dbh)) in enumerate(zip(paths, face_ids, db_dims)):
    script_lines.append(f'# face {fid}')
    script_lines.append(f'try:')
    script_lines.append(f'    img = Image.open("{fp}")')
    script_lines.append(f'    raw = img.size')
    script_lines.append(f'    orient = img.getexif().get(0x0112, "none")')
    script_lines.append(f'    img2 = ImageOps.exif_transpose(img)')
    script_lines.append(f'    print(f"face={fid:>6} raw={{raw}} exif_tag={{orient}} transposed={{img2.size}} DB={dbw}x{dbh} match={{img2.size == ({dbw},{dbh})}}")')
    script_lines.append(f'except Exception as e:')
    script_lines.append(f'    print(f"face={fid:>6} ERROR: {{e}}")')
    script_lines.append('')

script = '\n'.join(script_lines)

sftp = ssh.open_sftp()
with sftp.file('/tmp/test_dims.py', 'w') as f:
    f.write(script)
sftp.close()

cmd2 = 'docker cp /tmp/test_dims.py smart_photo_api:/tmp/ && docker exec smart_photo_api python3 /tmp/test_dims.py'
stdin2, stdout2, stderr2 = ssh.exec_command(cmd2)
print("\n=== PIL test ===")
print(stdout2.read().decode())
err = stderr2.read().decode().strip()
if err:
    print("ERR:", err[:300])

ssh.close()
