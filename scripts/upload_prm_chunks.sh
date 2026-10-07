#!/usr/bin/env bash
# Robust chunk uploader: append-only resume with md5-verified chunks.
# Invariants that matter on this flaky link:
#   - a failed `stat` is retried, never treated as "size 0" (that duplicated a chunk once)
#   - a chunk is accepted only when its remote md5 matches the local one
#   - oversize/mismatched chunks are deleted and re-sent from scratch
set -u
DEST=/datac/transfer_stage/prm_upload
SSHOPTS="-o ConnectTimeout=25 -o ServerAliveInterval=10 -o ServerAliveCountMax=4"
ssh $SSHOPTS wxh-server "mkdir -p $DEST"
for f in /tmp/prm_chunks/p*; do
  name=$(basename "$f")
  size=$(stat -c %s "$f")
  lmd5=$(md5sum "$f" | cut -d' ' -f1)
  while :; do
    out=$(ssh $SSHOPTS wxh-server "if [ -f $DEST/$name ]; then stat -c %s $DEST/$name; else echo MISSING; fi" 2>/dev/null) || { echo "stat-retry $name"; sleep 5; continue; }
    case "$out" in
      MISSING) rsize=0 ;;
      ''|*[!0-9]*) echo "stat-weird $name ($out)"; sleep 5; continue ;;
      *) rsize=$out ;;
    esac
    if [ "$rsize" -gt "$size" ]; then
      echo "oversize $name ($rsize) -> reset"; ssh $SSHOPTS wxh-server "rm -f $DEST/$name" || true; sleep 3; continue
    fi
    if [ "$rsize" -eq "$size" ]; then
      rmd5=$(ssh $SSHOPTS wxh-server "md5sum $DEST/$name | cut -d' ' -f1" 2>/dev/null) || { echo "md5-retry $name"; sleep 5; continue; }
      if [ "$rmd5" = "$lmd5" ]; then echo "OK   $name ($size md5 $rmd5)"; break; fi
      echo "md5 mismatch $name -> reset"; ssh $SSHOPTS wxh-server "rm -f $DEST/$name" || true; sleep 3; continue
    fi
    echo "xfer $name from $rsize/$size"
    tail -c +$((rsize + 1)) "$f" | ssh $SSHOPTS wxh-server "cat >> $DEST/$name" 2>/dev/null || true
    sleep 2
  done
done
ssh $SSHOPTS wxh-server "cd $DEST && cat p* > model.safetensors && stat -c %s model.safetensors && md5sum model.safetensors && rm -f p*"
echo "local md5: 26d4db719571b321952bf4920e319dcc"
echo "UPLOAD DONE"
