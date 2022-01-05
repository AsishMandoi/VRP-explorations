echo "Presenting $1..."

jupyter nbconvert $1 --to slides --post serve --SlidesExporter.reveal_scroll=True