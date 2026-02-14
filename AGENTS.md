# Agent Notes

- The control environment typically already has an SSH public key generated and added to the target server `47.120.50.21` in `~/.ssh/authorized_keys`.
- When accessing server data, use SSH to reach `47.120.50.21` and rely on the existing key on this control environment.
- Before deleting anything under `records`, run this pre-command first to write a JSON manifest to `/root/projects/Advanced_XLight/summary/exports`:

```bash
TS=$(date +%m%d_%H%M%S) && OUT="/root/projects/Advanced_XLight/summary/exports/records_predelete_manifest_${TS}.json" && python -c 'import os,json,sys,time; root="/root/projects/Advanced_XLight/records"; out=sys.argv[1]; items=[{"name":n,"path":os.path.join(root,n),"is_dir":os.path.isdir(os.path.join(root,n)),"size_bytes":(os.path.getsize(os.path.join(root,n)) if os.path.isfile(os.path.join(root,n)) else None),"mtime":os.path.getmtime(os.path.join(root,n))} for n in sorted(os.listdir(root))] if os.path.isdir(root) else []; data={"timestamp":time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),"records_root":root,"count":len(items),"entries":items}; os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(data, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=2); print(out)' "$OUT"
```
