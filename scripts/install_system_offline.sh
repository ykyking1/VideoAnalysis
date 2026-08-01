#!/usr/bin/env bash
# Hedef (internetsiz) makinede, scripts/setup.sh'DEN ONCE calistirilir - o,
# Python'a ihtiyac duyuyor, bu script Python OLMADAN da calisir (saf bash +
# dpkg/tar). scripts/prepare_offline_bundle.sh'in indirdigi .deb paketlerini
# ve Docker statik ikililerini kurar.
#
#     sudo ./scripts/install_system_offline.sh /yol/offline_bundle
#
# Idempotent - zaten kurulu olan (python3/git/ffmpeg/docker) atlanir.
set -euo pipefail

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[31m[HATA] %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" = "0" ] || die "root ile calistirin (sudo $0 <bundle>)."

BUNDLE="${1:-}"
[ -n "$BUNDLE" ] && [ -d "$BUNDLE/system_packages" ] || \
    die "Kullanim: sudo $0 /yol/offline_bundle  (once scripts/prepare_offline_bundle.sh calistirilmis olmali)"
BUNDLE="$(cd "$BUNDLE" && pwd)"

# --- python3 / git / ffmpeg (.deb paketlerinden) ----------------------------
say "Sistem paketleri (.deb) kuruluyor"
DEBS=("$BUNDLE"/system_packages/*.deb)
if [ -e "${DEBS[0]}" ]; then
    # `apt-get install ./*.deb` dpkg -i'den daha iyi: paketler arasi
    # bagimliligi kendisi sirali cozer, internete cikmaya CALISMAZ (tum
    # paketler zaten yerel dosya olarak veriliyor).
    apt-get install -y "${DEBS[@]}" || {
        warn "apt-get ile kurulum basarisiz, dpkg -i deneniyor (bagimlilik sirasi manuel olabilir)..."
        dpkg -i "${DEBS[@]}" || die ".deb paketleri kurulamadi - $BUNDLE/system_packages icindeki dosyalari kontrol edin."
    }
else
    warn "$BUNDLE/system_packages icinde .deb yok - python3/git/ffmpeg zaten kurulu oldugu varsayiliyor."
fi

for c in python3 git ffmpeg; do
    command -v "$c" >/dev/null 2>&1 || warn "$c hala bulunamiyor - .deb bundle'i eksik/uyumsuz olabilir (bkz. prepare_offline_bundle.sh'teki Ubuntu surumu uyarisi)."
done

# --- Docker (statik ikililerden, zaten kurulu degilse) ----------------------
if command -v docker >/dev/null 2>&1; then
    say "Docker zaten kurulu, atlaniyor: $(docker --version)"
else
    say "Docker statik ikililerinden kuruluyor"
    TGZ="$BUNDLE/system_packages/docker-static.tgz"
    [ -f "$TGZ" ] || die "$TGZ yok - prepare_offline_bundle.sh Docker'i indiremedi mi? Elle indirip buraya koyun."
    tmp="$(mktemp -d)"
    tar xzf "$TGZ" -C "$tmp"
    cp "$tmp"/docker/* /usr/bin/
    rm -rf "$tmp"

    # Minimal systemd servisi - resmi docker-ce paketinin kurdugu birim
    # dosyasinin sadelestirilmis hali.
    cat > /etc/systemd/system/docker.service <<'UNIT'
[Unit]
Description=Docker Application Container Engine
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/dockerd
Restart=always

[Install]
WantedBy=multi-user.target
UNIT
    groupadd -f docker
    systemctl daemon-reload
    systemctl enable --now docker
    usermod -aG docker "${SUDO_USER:-$USER}"
    warn "Docker grubuna eklendi - degisikligin etkili olmasi icin oturumu kapatip acin (ya da 'newgrp docker')."
    echo "Docker kuruldu: $(docker --version)"
fi

# docker-compose (v2 plugin) - statik dockerd tarball'inda YOK, ayri kurulur.
# Zaten kuruluysa (ör. distro'nun docker-ce-cli paketiyle geldiyse) atlanir.
if docker compose version >/dev/null 2>&1; then
    say "docker compose zaten kurulu, atlaniyor: $(docker compose version)"
else
    say "docker compose (v2) eklentisi kuruluyor"
    SRC="$BUNDLE/system_packages/docker-compose-plugin"
    if [ -f "$SRC" ]; then
        mkdir -p /usr/local/lib/docker/cli-plugins
        cp "$SRC" /usr/local/lib/docker/cli-plugins/docker-compose
        chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
        docker compose version >/dev/null 2>&1 \
            && echo "docker compose kuruldu: $(docker compose version)" \
            || warn "docker-compose kopyalandi ama 'docker compose version' hala calismiyor - elle kontrol edin."
    else
        warn "$SRC yok - docker compose eklentisi bundle'da bulunamadi. Elle indirin: https://github.com/docker/compose/releases (dosyayi /usr/local/lib/docker/cli-plugins/docker-compose olarak koyup +x yapin)."
    fi
fi

say "Tamam"
echo "Simdi normal kullanicidan (root DEGIL): ./scripts/setup.sh --offline $BUNDLE"
