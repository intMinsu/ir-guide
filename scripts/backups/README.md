[ [English](#english) | [한국어](#한국어) ]

# Backups & Restore (Official L4T)

**Up one level:** [`../README.md`](../README.md) · **Root:** [`../../README.md`](../../README.md)

---

## English

We use NVIDIA’s official tool under **Linux_for_Tegra/tools/backup_restore/** via this wrapper:

- `scripts/backups/jetson_backup.sh` (run on **host PC** inside `Linux_for_Tegra/`)

### Requirements
- Same **Linux_for_Tegra (L4T)** version as device (JetPack **5.1.5** → r35.6.x)
- Exactly **one** Jetson connected in **Force-Recovery** via USB
- For NVMe root: disk is usually `nvme0n1`

### Commands (host PC)
```bash
# 1) Go to Linux_for_Tegra (where flash.sh is)
cd ~/nvidia/nvidia_sdk/JetPack_5.1.5_*/Linux_for_Tegra

# 2) Make wrapper executable (assuming repo is cloned alongside)
chmod +x ../../scripts/backups/jetson_backup.sh

# 3) List images
../../scripts/backups/jetson_backup.sh list

# 4) Backup (NVMe root)
../../scripts/backups/jetson_backup.sh backup --ext-dev nvme0n1

# 5) Restore latest backup to NVMe
../../scripts/backups/jetson_backup.sh restore --ext-dev nvme0n1
```

> Tip: Restore uses the **latest** file in `tools/backup_restore/images/`.  
> Need a specific one? Temporarily move others out of the folder.

---

## 한국어

공식 **Linux_for_Tegra/tools/backup_restore/** 스크립트를 래핑한 툴을 사용합니다.

- 실행 파일: `scripts/backups/jetson_backup.sh` (호스트 PC, `Linux_for_Tegra/` 내부에서 실행)

### 준비물
- 장치와 동일한 **L4T 버전**(JetPack **5.1.5 → r35.6.x**)  
- USB로 연결된 **하나의** Jetson (Force-Recovery 모드)  
- NVMe 루트 사용 시 디스크 명(`nvme0n1`) 확인

### 사용법 (호스트 PC)
```bash
cd ~/nvidia/nvidia_sdk/JetPack_5.1.5_*/Linux_for_Tegra
chmod +x ../../scripts/backups/jetson_backup.sh

../../scripts/backups/jetson_backup.sh list
../../scripts/backups/jetson_backup.sh backup --ext-dev nvme0n1
../../scripts/backups/jetson_backup.sh restore --ext-dev nvme0n1
```

> 참고: 복원은 기본적으로 `tools/backup_restore/images/`의 **가장 최신** 백업을 사용합니다.  
> 특정 파일을 쓰려면 원하는 파일만 남기고 나머지는 임시로 이동하세요.

---

_Back to:_ **[Scripts](../README.md)** · **[Root README](../../README.md)** · **[Top](#backups--restore-official-l4t)**
