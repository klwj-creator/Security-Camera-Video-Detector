"""Build a two-page PDF explaining the current staff-tag prototype.

Run with the bundled PDF runtime, which supplies reportlab and Pillow.
"""

import argparse
import json
from pathlib import Path


def main():
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph
    from PIL import Image
    import reportlab

    # Embed bundled fonts so rendering does not depend on host fontconfig.
    fonts=Path(reportlab.__file__).parent/"fonts"
    pdfmetrics.registerFont(TTFont("Helvetica",str(fonts/"Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Helvetica-Bold",str(fonts/"VeraBd.ttf")))
    pdfmetrics.registerFontFamily("Helvetica",normal="Helvetica",bold="Helvetica-Bold")

    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--output",default="output/pdf/staff-detection-solution.pdf")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = Path(args.results)
    summary = json.loads((results/"summary.json").read_text())
    rows = json.loads((results/"trajectories.json").read_text())
    stills = json.loads((results/"evidence-stills.json").read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    pdf = canvas.Canvas(str(output),pagesize=A4)
    pdf.setTitle("Staff-tag detection: prototype solution and evaluation")
    pdf.setAuthor("Security Camera Video Detector project")
    W,H = A4
    margin = 43
    ink,muted,accent = HexColor("#17232D"),HexColor("#546371"),HexColor("#086A72")
    body = ParagraphStyle("body",fontName="Helvetica",fontSize=9.5,leading=14,textColor=ink)
    small = ParagraphStyle("small",parent=body,fontSize=8,leading=11,textColor=muted)
    y = H-45

    def paragraph(text,style=body,gap=9):
        nonlocal y
        p=Paragraph(text,style)
        _,height=p.wrap(W-2*margin,H)
        p.drawOn(pdf,margin,y-height)
        y-=height+gap

    def heading(text):
        nonlocal y
        y-=6
        pdf.setFillColor(accent)
        pdf.setFont("Helvetica-Bold",11)
        pdf.drawString(margin,y,text)
        y-=20

    def footer(page):
        pdf.setStrokeColor(HexColor("#D7DEE2"))
        pdf.line(margin,38,W-margin,38)
        pdf.setFont("Helvetica",8)
        pdf.setFillColor(muted)
        pdf.drawString(margin,24,"Development prototype | 14 November 2026 | One supplied recording")
        pdf.drawRightString(W-margin,24,str(page))

    pdf.setFillColor(ink)
    pdf.setFont("Helvetica-Bold",20)
    pdf.drawString(margin,y,"Staff-tag detection")
    y-=24
    paragraph("Prototype solution, reproducible code, and limits",ParagraphStyle("subtitle",parent=body,fontSize=11,textColor=muted),gap=13)
    heading("Task and interpretation")
    paragraph("Identify staff wearing the illustrated three-bar name tag in the supplied overhead video. "
              "The input contains <b>1,341 frames at 25 FPS, 960 x 720 pixels</b> (53.64 seconds). "
              "Frames in the outputs are zero-based; the CSV also supplies one-based frame numbers. "
              "A detected tag is evidence for a staff wearer. A missing tag detection is not proof of staff absence.")
    heading("Method")
    paragraph("<b>1. Detect the tag.</b> Fine-tune YOLO11 nano on reviewed tag boxes from person crops. "
              "Train with rotation and vertical-flip augmentation for the ceiling camera. Include plain clothes, "
              "patterned shirts, unrelated shirt lettering, and scenery as negatives. Verify the local three-bar "
              "shape against scaled and rotated reference templates; correlation scores are not probabilities.")
    paragraph("<b>2. Check person context.</b> Run pretrained person detection on the original and 180-degree view, "
              "map boxes back to original coordinates, and suppress overlapping proposals. Reject tag predictions "
              "outside detected people or implausibly large relative to the containing body. This reduces furniture lookalikes.")
    paragraph("<b>3. Confirm short trajectories.</b> Require at least two tag observations associated by bounded "
              "image motion. Interpolate gaps of at most five frames (0.20 seconds), without extrapolating before "
              "the first or after the last observation. Mark direct and interpolated evidence separately. "
              "Long-lived person tracker IDs are deliberately not treated as identities.")
    paragraph("<b>4. Export every frame.</b> Write a presence prediction, evidence type, xy coordinates, "
              "and track IDs to CSV, plus an annotated video and the underlying detections for review.")
    heading("The visual cue")
    # Supplied references illustrate the target without hiding its bars behind overlays.
    image_top=y
    examples=[root/"details/reference-images/reference-2-Image2.jpg",root/"details/reference-images/reference-3-Image3.jpg"]
    for i,path in enumerate(examples):
        image=Image.open(path)
        h=135
        w=h*image.width/image.height
        x=margin+48+i*245
        pdf.drawImage(ImageReader(image),x,image_top-h,width=w,height=h,preserveAspectRatio=True)
        pdf.setFont("Helvetica",8)
        pdf.setFillColor(muted)
        pdf.drawString(x,image_top-h-14,"Supplied dark-shirt example" if i==0 else "Supplied light-shirt example")
    y=image_top-163
    paragraph("Source: AI Evaluation Test.pdf and sample.mp4, preserved in details/. "
              "The tag is a small black-and-white pattern; clothing color alone is not the staff rule.",small)
    if y < 50:
        raise RuntimeError("Page 1 overflow; revise layout.")
    footer(1)
    pdf.showPage()
    y=H-46
    pdf.setFont("Helvetica-Bold",17)
    pdf.setFillColor(ink)
    pdf.drawString(margin,y,"Results and reproducibility")
    y-=27
    heading("Current sample output")
    paragraph(f"The current automatic run predicts staff-tag presence in <b>{summary['predicted_present_frames']} / "
              f"{summary['total_frames']} frames</b>: {summary['direct_frames']} contain retained direct detections "
              f"and {summary['interpolated_only_frames']} are short-gap interpolation only. These are <b>model "
              "predictions, not exhaustively verified staff-presence ground truth</b>. Exact predicted intervals "
              "are in summary.json; all frame decisions are in frame-results.csv.")
    paragraph("<b>Coordinates:</b> staff_xy is the center of the detected tag, used as a chest-location proxy "
              "in image pixels: origin at top left, x increases right, y increases down. An associated person-box "
              "center is supplied separately where available. Neither is a calibrated floor coordinate.")
    heading("Training and validation scope")
    paragraph("The refined development dataset contains 15 positive + 52 negative training crops, "
              "15 positive + 44 negative validation crops, and 5 positive + 56 negative test crops. "
              "Positive examples are grouped by separate walking passes rather than adjacent frames across splits. "
              "All come from <b>one recording, camera, and scene</b>; the test set is too small and correlated "
              "to establish deployment accuracy. Full-clip visual review was used for hard-negative mining.")
    paragraph("The software checks cover decoding, original-coordinate mapping, tag-pattern matching, "
              "confirmation, duplicate ownership, bounded interpolation, stale-track protection, and output "
              "writing. Thirty automated checks pass. Crop detection metrics are recorded with training/test "
              "logs; they must not be presented as whole-video staff-presence precision or recall.")
    heading("Run on another local video")
    paragraph("From the project folder, activate .venv, then run:")
    command="python scripts/detect_staff.py /path/to/video.mp4 --output outputs/new"
    paragraph(command,ParagraphStyle("code",parent=body,fontName="Helvetica",fontSize=8.2,leading=12),gap=10)
    paragraph("Default weights: models/staff-tag-sample-v2.pt. The person detector runs automatically for a new video. A person-cache option can accelerate "
              "repeat experiments only when the source video SHA-256 matches. The recorded sample timing with "
              "cached person boxes excludes person inference and is not an end-to-end real-time benchmark.")
    heading("Limits and next work")
    paragraph("The current output is a prototype. Blur, tiny tags, long occlusion, and turns that hide the "
              "badge can miss staff who are still present. Similar patterns can cause false positives. "
              "Resolve this with more reviewed tag examples, full per-frame presence annotations, and "
              "independent recordings from the intended camera. Measure false alerts, missed staff appearances, "
              "ID switches, and complete pipeline latency before using the system operationally.")
    paragraph("Files: scripts/detect_staff.py (entry point); cctv_detector/ (pipeline and tracking); "
              "models/ (local trained weights); outputs/ (CSV, JSON, annotated video); details/ (original sources). "
              "Official tool references: docs/sources.md. No external submission has been made.",small)
    if y < 50:
        raise RuntimeError("Page 2 overflow; revise layout.")
    footer(2)
    pdf.save()
    print(output)


if __name__ == "__main__":
    main()
