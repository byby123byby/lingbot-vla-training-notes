from pathlib import Path
import json, math, hashlib
import av
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

OUT = Path('/workspace/video_review_remaining33_20261024')
OUT.mkdir(exist_ok=True)
CASES = [('adjust_bottle', 16, 135), ('adjust_bottle', 20, 16), ('adjust_bottle', 25, 14), ('beat_block_hammer', 26, 15), ('blocks_ranking_size', 28, 321), ('move_stapler_pad', 1, 13), ('open_laptop', 11, 14), ('open_laptop', 19, 15), ('open_laptop', 34, 14), ('place_cans_plasticbox', 12, 264), ('place_fan', 13, 18), ('put_bottles_dustbin', 17, 436), ('put_bottles_dustbin', 43, 447), ('rotate_qrcode', 5, 96), ('rotate_qrcode', 19, 97), ('scan_object', 4, 18), ('scan_object', 20, 106), ('scan_object', 42, 110), ('shake_bottle_horizontally', 20, 19), ('shake_bottle_horizontally', 21, 116), ('shake_bottle_horizontally', 40, 15), ('shake_bottle_horizontally', 45, 110), ('shake_bottle', 21, 15), ('shake_bottle', 40, 14), ('shake_bottle', 45, 14), ('stamp_seal', 0, 21), ('turn_switch', 3, 41), ('turn_switch', 7, 55), ('turn_switch', 12, 41), ('turn_switch', 24, 53), ('turn_switch', 28, 41), ('turn_switch', 39, 41), ('turn_switch', 42, 48)]
records = []

def sheet(items, path, cols=5):
    canvas = Image.new('RGB', (320*cols, 260*math.ceil(len(items)/cols)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(items):
        x,y = (i%cols)*320,(i//cols)*260
        canvas.paste(im,(x,y+20));draw.text((x+3,y+3),label,fill='black')
    canvas.save(path,quality=90)

for task,ep,peak in CASES:
    p=Path('/RoboTwin/data/lerobot')/(task+'_joint_v30')
    info=json.loads((p/'meta/info.json').read_text())
    meta=[r for f in sorted(p.glob('meta/episodes/**/*.parquet')) for r in pq.read_table(f).to_pylist()]
    m=next(r for r in meta if r['episode_index']==ep)
    n=m['length'];fps=info['fps'];cams=[k for k,v in info['features'].items() if v['dtype']=='video']
    images={}; sources=[]
    for cam in cams:
        prefix='videos/'+cam+'/'
        path=p/info['video_path'].format(video_key=cam,chunk_index=m[prefix+'chunk_index'],file_index=m[prefix+'file_index'])
        start=m[prefix+'from_timestamp'];frames=[]
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            container.seek(int(start/float(stream.time_base)),stream=stream,backward=True)
            for frame in container.decode(stream):
                if frame.time is None or frame.time < start-.25/fps:continue
                expected=start+len(frames)/fps
                assert abs(frame.time-expected)<.25/fps,(task,ep,cam,frame.time,expected)
                frames.append(frame.to_image())
                if len(frames)==n:break
        assert len(frames)==n,(task,ep,cam,len(frames),n)
        images[cam]=frames
        sources.append(dict(camera=cam,path=str(path),start_timestamp=start,frames=n))
    stem=f'{task}_ep{ep:02d}'
    # Save the entire 3-camera episode, with original native frame rate and resolution.
    with av.open(str(OUT/(stem+'.mp4')),'w') as video:
        stream=video.add_stream('libx264',rate=fps)
        stream.width=960;stream.height=260;stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'fast'}
        for t in range(n):
            im=Image.new('RGB',(960,260),'white');d=ImageDraw.Draw(im)
            for j,cam in enumerate(cams):
                im.paste(images[cam][t],(j*320,20))
                d.text((j*320+2,2),f'{cam.rsplit(".",1)[-1]} frame {t}/{n-1}',fill='black')
            for packet in stream.encode(av.VideoFrame.from_image(im)):video.mux(packet)
        for packet in stream.encode():video.mux(packet)
    high=images['observation.images.cam_high']
    # Every third frame for the full episode; every frame around the numeric peak.
    overview=sorted(set(range(0,n,3))|{n-1})
    for page,lo in enumerate(range(0,len(overview),40)):
        ids=overview[lo:lo+40]
        sheet([(high[t],f'{stem} f{t}') for t in ids],OUT/(stem+f'_overview_{page}.jpg'))
    ids=list(range(max(0,peak-6),min(n,peak+10)))
    for page,lo in enumerate(range(0,len(ids),8)):
        # Each row contains all 3 cameras at one timestamp.
        cells=[(images[cam][t],f'f{t} {cam.rsplit(".",1)[-1]}') for t in ids[lo:lo+8] for cam in cams]
        sheet(cells,OUT/(stem+f'_peak_{page}.jpg'),cols=3)
    record=dict(task=task,episode=ep,peak_frame=peak,frames=n,fps=fps,instructions=m['tasks'],
                sources=sources,overview_frames=overview,peak_frames=ids,video=stem+'.mp4')
    records.append(record)
    print(stem,n,'frames per camera; complete video decoded',flush=True)
(OUT/'extraction_manifest.json').write_text(json.dumps(records,indent=2))
hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file()}
(OUT/'sha256.json').write_text(json.dumps(hashes,indent=2))
print('COMPLETE',len(records),sum(r['frames'] for r in records)*3,'camera frames',flush=True)
