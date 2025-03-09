if ! command -v unrar &> /dev/null; then
  apt update
  apt install unrar -y
fi

mkdir data/original
unrar x data/original.rar data/original
