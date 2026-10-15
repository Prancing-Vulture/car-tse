"""Rasterize this report's simple SVG charts with Pillow (no GUI backend)."""
from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SCALE = 2
FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")


def font(size: int):
    try:
        return ImageFont.truetype(str(FONT_PATH), max(8, int(size*SCALE)))
    except OSError:
        return ImageFont.load_default()


def render(path: Path):
    root = ET.parse(path).getroot()
    width = int(float(root.attrib["width"]))
    height = int(float(root.attrib["height"]))
    image = Image.new("RGB", (width*SCALE, height*SCALE), "white")
    draw = ImageDraw.Draw(image)
    css = {}
    style = root.find("{http://www.w3.org/2000/svg}style")
    if style is not None and style.text:
        for selector, body in re.findall(r"\.([\w-]+)\s*\{([^}]+)\}", style.text):
            css[selector] = {k.strip(): v.strip() for k,v in (part.split(":",1) for part in body.split(";") if ":" in part)}
    for node in root:
        tag = node.tag.rsplit("}",1)[-1]
        attrs = node.attrib
        klass = attrs.get("class", "")
        stylemap = css.get(klass, {})
        fill = attrs.get("fill", stylemap.get("fill", "#253247"))
        stroke = attrs.get("stroke", stylemap.get("stroke", "none"))
        sw = float(attrs.get("stroke-width", stylemap.get("stroke-width", 1))) * SCALE
        if tag == "rect":
            x,y=float(attrs.get("x",0)),float(attrs.get("y",0))
            w=width if attrs.get("width")=="100%" else float(attrs["width"])
            h=height if attrs.get("height")=="100%" else float(attrs["height"])
            draw.rectangle((x*SCALE,y*SCALE,(x+w)*SCALE,(y+h)*SCALE), fill=fill if fill!="none" else None, outline=stroke if stroke!="none" else None, width=max(1,int(sw)))
        elif tag == "line":
            xy=[float(attrs[k])*SCALE for k in ("x1","y1","x2","y2")]
            draw.line(xy, fill=stroke if stroke!="none" else fill, width=max(1,int(sw)))
        elif tag == "circle":
            x,y,r=(float(attrs[k])*SCALE for k in ("cx","cy","r"))
            draw.ellipse((x-r,y-r,x+r,y+r), fill=fill, outline=stroke if stroke!="none" else None, width=max(1,int(sw)))
        elif tag == "polyline":
            coords=[tuple(float(n)*SCALE for n in pair.split(",")) for pair in attrs.get("points","").split()]
            if len(coords)>1: draw.line(coords, fill=stroke if stroke!="none" else fill, width=max(1,int(sw)), joint="curve")
        elif tag == "text":
            text=node.text or ""; size=int(float(stylemap.get("font-size",14).replace("px", "")))
            f=font(size); x=float(attrs.get("x",0))*SCALE; y=float(attrs.get("y",0))*SCALE
            anchor=attrs.get("text-anchor", "start")
            box=draw.textbbox((0,0),text,font=f); tw=box[2]-box[0]
            if anchor=="middle": x-=tw/2
            elif anchor=="end": x-=tw
            # SVG y is the text baseline; Pillow positions text from the top.
            y-=size*SCALE*.82
            transform=attrs.get("transform","")
            rot=re.search(r"rotate\((-?[0-9.]+)\)", transform)
            trans=re.search(r"translate\(([0-9.]+)[ ,]+([0-9.]+)\)", transform)
            if trans:
                x,y=float(trans.group(1))*SCALE,float(trans.group(2))*SCALE
            if rot:
                layer=Image.new("RGBA",(max(1,tw+8*SCALE),size*SCALE*2),(255,255,255,0))
                ImageDraw.Draw(layer).text((4*SCALE,0),text,font=f,fill=fill)
                layer=layer.rotate(float(rot.group(1)),expand=True)
                image.paste(layer,(int(x-layer.width/2),int(y-layer.height/2)),layer)
            else:
                draw.text((x,y),text,font=f,fill=fill)
    image.resize((width,height),Image.Resampling.LANCZOS).save(path.with_suffix(".png"), optimize=True)


if __name__ == "__main__":
    files=sorted(HERE.glob("fig1[1-5]*.svg"))
    for file in files: render(file)
    print(f"Rendered {len(files)} PNGs in {HERE}")
