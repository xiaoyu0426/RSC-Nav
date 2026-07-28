# RSC-Nav Project Notes

## Credential Safety

Never place private-key contents, API keys, passphrases, or other credential
values in this repository, logs, manifests, commits, or user-facing output.

The authorized SSH key is stored locally in this archive:

```text
C:\Users\91954\xwechat_files\wxid_3yq27o6l0fu322_5a7c\msg\file\2026-06\.ssh(1).zip
```

The private-key entry is `.ssh(1)/id_ed25519`. Extract it only to a temporary
directory outside the repository, restrict its ACL to the current user, and
delete the extracted copy when it is no longer needed. On Windows, use `-F NUL`
to bypass the empty local SSH config if its ACL is rejected. Use a temporary
per-session `UserKnownHostsFile`; do not disable host-key checking globally.
After an expected development-instance restart, accept the new host key into
that temporary file. Treat an unexplained host-key change as a connection
failure.

## Primary Development Machine

The current GPU development instance is `dev-17`:

```text
InstanceId: dsw-wtl07y29d804h3d2ju
group_chat_id: oc_993b9eb51ae20e7dad39b698e812abfb
SSH: ssh -p 1004 yujiexiao@8.147.67.185
Public HTTP DNAT range: 39.101.65.229:44801-44900
```

Primary project and runtime paths:

```text
/workspace/yujiexiao/RSC_Nav
/workspace/yujiexiao/miniconda3/envs/rscnav-habitat22/bin/python
/workspace/yujiexiao/miniconda3/envs/lingbot-bench/bin/python
/workspace/yujiexiao/.rscnav/habitat_data/versioned_data/hm3d-0.2/hm3d/example
```

The instance has four NVIDIA A800 80GB GPUs when fully provisioned. Always
inspect GPU utilization before selecting a detector device. Instance restarts
terminate processes, clear `/tmp`, and may change the hostname and SSH host key;
the shared `/workspace` data normally persists.

Use the remote machine for Habitat traversal, RGB-D export, GroundingDINO,
RSC-Nav simulation runs, and heavier visualization generation. Before using
the existing project directory, inspect `git status`; it may contain unrelated
or generated work. Prefer an independent clean clone or worktree for a
reproducible run.

## Fallback Development Machine

The older endpoint remains available for file inspection and fallback work:

```text
SSH: ssh -p 1040 yujiexiao@39.101.65.229
Workspace: /workspace/yujiexiao/RSC_Nav
Legacy public HTTP range: 39.101.65.229:43000-44000
```

Do not assume that this endpoint currently has a visible GPU or the same
runtime symlinks. Check `nvidia-smi` and resolve the actual Conda path before
starting heavy work.

## Git And Transfer Policy

Remote development machines are strictly pull/read/run only:

- Never run `git push` from either remote machine.
- Push GitHub changes only from the local workspace.
- Do not overwrite or clean a dirty remote workspace.
- Use a clean remote clone for reproducible runs, then pull a known commit.
- Record the exact run commit in the run manifest.
- Transfer selected artifacts back to the local workspace before committing.

The dev-17 SFTP subsystem may reject modern `scp`; use legacy SCP mode (`scp
-O`) when needed. Never include credentials in a run manifest or command log.
