#!/usr/bin/env bash
# Resumable chunk uploader: each chunk is appended from the last verified
# remote byte (tail -c +N | ssh cat >>), so a dropped connection resumes
# instead of restarting. Chunks already at full size are skipped.
set -u
DEST=/datac/transfer_stage/prm_upload
SSHOPTS="-o ConnectTimeout=25 -o ServerAliveInterval=10 -o ServerAliveCountMax=4"
ssh $SSHOPTS wxh-server "mkdir -p $DEST"
for f in /tmp/prm_chunks/p*; do
  name=$(basename "$f")
  size=$(stat -c %s "$f")
  while :; do
    rsize=$(ssh $SSHOPTS wxh-server "stat -c %s $DEST/$name 2>/dev/null" 2>/dev/null || echo 0)
    [ -z "$rsize" ] && rsize=0
    if [ "$rsize" -ge "$size" ]; then echo "OK   $name ($rsize/$size)"; break; fi
    echo "xfer $name from $rsize/$size"
    tail -c +$((rsize + 1)) "$f" | ssh $SSHOPTS wxh-server "cat >> $DEST/$name" 2>/dev/null || true
    sleep 3
  done
done
ssh $SSHOPTS wxh-server "cd $DEST && cat p* > model.safetensors && stat -c %s model.safetensors && rm -f p*"
ssh $SSHOPTS wxh-server "md5sum $DEST/model.safetensors"
echo "local: 26d4db719571b321952bf4920e319dcc  (expected)"
echo "UPLOAD DONE"
