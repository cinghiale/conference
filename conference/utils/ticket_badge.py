#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import json
import logging
import math
import optparse
import os
import os.path
import re
import sys
from PIL import Image, ImageDraw, ImageFont, ImageMath
from itertools import izip_longest

parser = optparse.OptionParser(usage='%(prog)s [options] output_dir')
parser.add_option("-i", "--input",
                  dest="input",
                  default=None,
                  action="store",
                  help="input file (default stdin)")
parser.add_option("--verbose",
                  dest="verbose",
                  default=False,
                  action="store_true",
                  help="be verbose")
parser.add_option("-p", "--page-size",
                  dest="page_size",
                  default="490x318",
                  action="store",
                  help="page size (mm)")
parser.add_option("-d", "--dpi",
                  dest="dpi",
                  default=300,
                  action="store",
                  type="int",
                  help="dpi")
parser.add_option("-r", "--resize",
                  dest="resize",
                  default=None,
                  action="store",
                  type="float",
                  help="resize factor (if any)")
parser.add_option("-n", "--per-page",
                  dest="per_page",
                  default=9,
                  action="store",
                  type="int",
                  help="badge per page")
parser.add_option("-c", "--conf",
                  dest="conf",
                  default="conf.py",
                  action="store",
                  help="configuration script")
parser.add_option("-e", "--empty-pages",
                  dest="empty_pages",
                  default="0",
                  action="store",
                  help="prepare x empty pages")
parser.add_option("--center",
                  dest="align_center",
                  default=False,
                  action="store_true",
                  help="align badges horizontally")
parser.add_option("--x-mirror",
                  dest="mirror_x",
                  default=False,
                  action="store_true",
                  help="reorder badge along the x axis")

opts, args = parser.parse_args()

try:
    output_dir = args[0]
except IndexError:
    parser.print_usage()


MM2INCH = 0.03937


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def wrap_text(font, text, width):
    words = re.split(' ', text)
    lines = []
    while words:
        word = words.pop(0).strip()
        if not word:
            continue
        if not lines:
            lines.append(word)
        else:
            line = lines[-1]
            w, h = font.getsize(line + ' ' + word)
            if w <= width:
                lines[-1] += ' ' + word
            else:
                lines.append(word)

    for ix, line in enumerate(lines):
        line = line.strip()
        while True:
            w, h = font.getsize(line)
            if w <= width:
                break
            line = line[:-1]
        lines[ix] = line
    return lines


def draw_info(image, max_width, text, pos, font, color, line_offset=8):
    d = ImageDraw.Draw(image)

    lowline_check = 'gjqpy'

    cx = pos[0]
    cy = pos[1] - font.getsize(text)[1]
    if set(lowline_check) & set(text):
        diff = font.getsize('g')[1] - font.getsize('o')[1]
        cy += diff
        line_offset += diff
    lines = wrap_text(font, text, max_width)
    for l in lines:
        d.text((cx, cy), l, font=font, fill=color)
        cy += font.getsize(l)[1] + line_offset

    return len(lines), cy


def open_font(file_path, points, _cache={}):
    """
    Open a truetype font and set the size to the specified in points.
    """
    key = (file_path, points)
    try:
        return _cache[key]
    except KeyError:
        f = _cache[key] = ImageFont.truetype(file_path, points * DPI/72)
        return f


def ticket_group(ticket):
    try:
        return ticket["_ticket_group"]
    except:
        return ""


# http://stackoverflow.com/questions/765736/using-pil-to-make-all-white-pixels-transparent#answer-4531395
def distance2(a, b):
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])


def makeColorTransparent(image, color, thresh2=0):
    image = image.convert("RGBA")
    red, green, blue, alpha = image.split()

    prg = ImageMath.eval(
        """convert(((((t - d(c, (r, g, b))) >> 31) + 1) ^ 1) * a, 'L')""",
        t=thresh2, d=distance2, c=color, r=red, g=green, b=blue, a=alpha)
    image.putalpha(prg)
    return image


def split_name(name):
    """
    Split a name in two pieces: first_name, last_name.
    """
    parts = name.split(' ')
    if len(parts) == 4 and parts[2].lower() not in ('de', 'van'):
        first_name = ' '.join(parts[:2])
        last_name = ' '.join(parts[2:])
    else:
        first_name = parts[0]
        last_name = ' '.join(parts[1:])
    return first_name.strip(), last_name.strip()


def open_auxiliary_image(path, mm, transparent_pixel=(0, 0), threshold=150):
    img = Image.open(path)
    if img.mode != 'RGBA':
        if img.mode == 'LA':
            img = img.convert('RGBA')
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if transparent_pixel:
                img = makeColorTransparent(
                    img,
                    img.getpixel(transparent_pixel),
                    thresh2=threshold)
    size = map(int, (mm * MM2INCH * DPI, mm * MM2INCH * DPI))
    return img.resize(size, Image.ANTIALIAS)


util_functions = {
    'wrap_text': wrap_text,
    'draw_info': draw_info,
    'open_font': open_font,
    'make_color_transparent': makeColorTransparent,
    'split_name': split_name,
    'ticket_group': ticket_group,
    'open_auxiliary_image': open_auxiliary_image,
}


class RenderProcess(object):
    def __init__(self,
                 output_dir,
                 render_ticket,
                 resize_factor=0,
                 empty_pages=0):
        self.output_dir = output_dir
        self.render_ticket = render_ticket

        self.opts = {
            "resize_factor": resize_factor,
            "empty_pages": empty_pages,
        }

        self.group_id = None
        self.group_data = None

        self.counter = 0
        self.pages = 0

        self.empty_counter = 0
        self.empty_pages = 0

    @property
    def attendees(self):
        return self.group_data["attendees"]

    @property
    def image(self):
        return self.group_data["image"]

    def _write_page(self, name, page):
        logging.debug('writing %s', name)
        with file(os.path.join(self.output_dir, name), 'w') as out:
            page.save(out, 'TIFF', dpi=(DPI, DPI))

    def _resize_image(self, image, factor):
        nsize = image.size[0] * factor, image.size[1] * factor
        return image.resize(nsize, Image.ANTIALIAS)

    def _save_page(self, page):
        self.counter += 1

        tpl = "[{group}] pag {counter}-{total}.tif"
        file_name = tpl.format(group=self.group_id,
                               counter=self.counter,
                               total=self.pages)

        self._write_page(file_name, page)

    def _save_empty_page(self, page):
        self.empty_counter += 1

        tpl = "[{group}][blank] pag {counter}-{total}.tif"
        file_name = tpl.format(group=self.group_id,
                               counter=self.empty_counter,
                               total=self.empty_pages)

        self._write_page(file_name, page)

    def _render_single_page(self, attendees):
        images = []
        for attendee in attendees:
            if attendee:
                # a None value is used to pad the page
                try:
                    attendee['_ticket_group'] = self.group_id
                except TypeError:
                    # attendee cannot be altered (like a tuple or a frozen
                    # class/dict)
                    pass

            badge = self.render_ticket(self.image,
                                       attendee,
                                       utils=util_functions)
            images.append(badge)
        if self.opts["resize_factor"]:
            images = map(
                lambda i: self._resize_image(i, self.opts["resize_factor"]),
                images)
        return images

    def _tickets_number(self):
        return len(self.attendees)

    def _ticket_pages(self):
        raise NotImplementedError()

    def _render_ticket_pages(self):
        raise NotImplementedError()

    def _render_empty_pages(self):
        raise NotImplementedError()

    def _empty_pages(self):
        v = self.opts["empty_pages"]
        if isinstance(v, basestring) and v.endswith('%'):
            t = math.ceil(self._ticket_pages() * float(v[:-1]) / 100)
            p = int(t)
        else:
            p = int(v)
        return p

    def run(self, group_id, group_data):
        self.group_id = group_id
        self.group_data = group_data

        self.counter = 0
        self.pages = self._ticket_pages()

        self.empty_counter = 0
        self.empty_pages = self._empty_pages()

        logging.info("processing %s, %d tickets %d pages %d empty pages",
                     self.group_id,
                     self._tickets_number(),
                     self.pages,
                     self.empty_pages)

        if len(self.attendees):
            self._render_ticket_pages()

        if self.empty_pages > 0:
            self._render_empty_pages()


class SingleTicketRender(RenderProcess):
    def __init__(self, *args, **kw):
        align = kw.pop('align')
        images_per_page = kw.pop('images_per_page')
        super(SingleTicketRender, self).__init__(*args, **kw)
        self.opts.update({
            "align": align,
            "images_per_page": images_per_page,
        })

    def _assemble_page(self, images):
        page = Image.new('RGBA', PAGE_SIZE, (255, 255, 255, 255))
        limits = (
            PAGE_SIZE[0] - 2*PAGE_MARGIN,
            PAGE_SIZE[1] - 2*PAGE_MARGIN)

        x = y = 0
        rows = [[]]
        for img in images:
            size = img.size
            if x + size[0] > limits[0]:
                x = 0
                y += size[1]
                rows.append([])
            rows[-1].append((img, (x, y)))
            x += size[0]

        align = self.opts["align"]
        mirror_x = self.group_data.get('mirror_x', False)
        for row in rows:
            if align == 'center':
                align_offset = 1
                row_width = sum([])
                for img, pos in row:
                    row_width += img.size[0]
                align_offset = (PAGE_SIZE[0] - row_width) / 2, PAGE_MARGIN
            else:
                align_offset = PAGE_MARGIN, PAGE_MARGIN
            if mirror_x:
                original = row
                mirrored = row[::-1]
                for ix, el in enumerate(zip(original, mirrored)):
                    img = el[0][0]
                    mirrored_pos = el[1][1]
                    row[ix] = (img, mirrored_pos)
            for img, pos in row:
                x, y = pos
                align_x, align_y = align_offset
                page.paste(img, (x + align_x, y + align_y), img)
        return page

    def _render_ticket_pages(self):
        for block in grouper(self.opts["images_per_page"], self.attendees):
            images = self._render_single_page(block)
            self._save_page(self._assemble_page(images))

    def _render_empty_pages(self):
        images = self._render_single_page(
            [None] * self.opts["images_per_page"])

        for ix in range(self.empty_pages):
            self._save_empty_page(self._assemble_page(images))

    def _ticket_pages(self):
        n = self.opts["images_per_page"]
        p = len(self.attendees) / n
        if len(self.attendees) % n:
            p += 1
        return p


class WholePageRender(RenderProcess):
    def __init__(self, *args, **kw):
        tickets_per_page = kw.pop('tickets_per_page')
        super(WholePageRender, self).__init__(*args, **kw)
        self.opts.update({
            "tickets_per_page": tickets_per_page,
        })

    def _tickets_number(self):
        return len(self.attendees) * self.opts["tickets_per_page"]

    def _ticket_pages(self):
        # every item in self.attendees is a gruop handled by the
        # render_ticket function
        return len(self.attendees)

    def _render_ticket_pages(self):
        for slice in self.attendees:
            images = self._render_single_page([slice])
            self._save_page(images[0])

    def _render_empty_pages(self):
        images = self._render_single_page([[]])
        for ix in range(self.empty_pages):
            self._save_empty_page(images[0])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG if opts.verbose else logging.WARN)

    badge_align = 'left' if not opts.align_center else 'center'
    badge_x_mirror = opts.mirror_x

    logging.info("executing %s", opts.conf)
    conf = {}
    os.chdir(os.path.dirname(opts.conf))
    execfile(os.path.basename(opts.conf), conf)

    try:
        group_tickets = conf['tickets']
        render_ticket = conf['ticket']
    except KeyError, e:
        logging.error("missing config key: %s", e)
        sys.exit(1)

    DPI = opts.dpi
    WASTE = conf.get('WASTE', 0) * MM2INCH * DPI
    PAGE_MARGIN = int(conf.get('PAGE_MARGIN', 10) * MM2INCH * DPI)

    RENDER_MODE = conf.get("RENDER_MODE", "single-ticket")
    logging.info("render mode: %s", RENDER_MODE)
    if RENDER_MODE not in ("single-ticket", "whole-page"):
        logging.error("render mode not supported")
        sys.exit(1)

    if opts.page_size == 'A3':
        psize = "420x297"
    elif opts.page_size == 'A4':
        psize = "297x210"
    else:
        psize = opts.page_size

    try:
        PAGE_SIZE = map(lambda x: int(int(x) * MM2INCH * DPI),
                        psize.split('x'))
    except Exception:
        logging.error("invalid page size")
        sys.exit(1)
    logging.info("page size: %sx%s", *PAGE_SIZE)

    data = json.loads(sys.stdin.read())
    groups = group_tickets(data)
    logging.info("%d tickets divided into %d groups: %s",
                 len(data),
                 len(groups),
                 ",".join(groups.keys()))

    if RENDER_MODE == "single-ticket":
        proc = SingleTicketRender(
            output_dir,
            render_ticket,
            resize_factor=opts.resize,
            empty_pages=opts.empty_pages,
            align=badge_align,
            images_per_page=opts.per_page,
        )
    else:
        proc = WholePageRender(
            output_dir,
            render_ticket,
            resize_factor=opts.resize,
            empty_pages=opts.empty_pages,
            tickets_per_page=conf["TICKETS_PER_PAGE"],
        )

    for (name, group) in sorted(groups.items()):
        group.setdefault("mirror_x", badge_x_mirror)
        proc.run(group_id=name, group_data=group)
