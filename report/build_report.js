// Builds deliverables/Scaffold_Plank_Counter_Report.docx from the evaluation outputs.
// Run: node report/build_report.js   (after plank-count classical / yolo / coreml)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  HeadingLevel, WidthType, ShadingType, BorderStyle, LevelFormat, Footer, PageNumber,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "outputs");
const FONT = "Arial";
const NAVY = "1F3A5F";
const PAGE_W = 9026; // A4 width minus 1" margins, in DXA

// ---------- data ----------
const IMAGES = ["plank1", "plank2", "plank3", "plank4", "plank5"];
const TRUE = { plank1: 54, plank2: 27, plank3: 76, plank4: 18, plank5: 60 };

const parseTable = (file) =>
  Object.fromEntries(
    fs.readFileSync(path.join(OUT, file), "utf8").split("\n")
      .filter((l) => l.startsWith("| plank"))
      .map((l) => l.split("|").map((c) => c.trim()))
      .map((c) => [c[1], { predicted: +c[2], ms: +c[5] }]),
  );
const classical = parseTable("classical.md");
const yolo = parseTable("yolo.md");
const coreml = fs.readFileSync(path.join(OUT, "coreml.md"), "utf8");
const [, sizeMb] = coreml.match(/([\d.]+) MB/);
const [, medianMs] = coreml.match(/median ([\d.]+) ms/);
const [, p90Ms] = coreml.match(/p90 ([\d.]+) ms/);

const total = (m) => IMAGES.reduce((s, n) => s + Math.abs(m[n].predicted - TRUE[n]), 0);
const trueTotal = IMAGES.reduce((s, n) => s + TRUE[n], 0);
const accuracy = (m) => (100 * (1 - total(m) / trueTotal)).toFixed(1);
const mae = (m) => (total(m) / IMAGES.length).toFixed(1);
const meanMs = (m) => (IMAGES.reduce((s, n) => s + m[n].ms, 0) / IMAGES.length).toFixed(0);
const signed = (n) => (n > 0 ? `+${n}` : `${n}`);

// ---------- building blocks ----------
const run = (text, opts = {}) => new TextRun({ text, font: FONT, ...opts });
const para = (children, opts = {}) =>
  new Paragraph({ children: typeof children === "string" ? [run(children)] : children, spacing: { after: 120 }, ...opts });
const rich = (parts) => para(parts.map((p) => (typeof p === "string" ? run(p) : run(p[0], p[1]))));
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [run(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [run(text)] });
const bullet = (parts) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 },
    children: (Array.isArray(parts) ? parts : [parts]).map((p) => (typeof p === "string" ? run(p) : run(p[0], p[1]))),
  });
const numbered = (parts, instance = 1) =>
  new Paragraph({
    numbering: { reference: "steps", level: 0, instance },
    spacing: { after: 60 },
    children: (Array.isArray(parts) ? parts : [parts]).map((p) => (typeof p === "string" ? run(p) : run(p[0], p[1]))),
  });

const border = { style: BorderStyle.SINGLE, size: 4, color: "C9D2DC" };
const cell = (text, width, { header = false, align = AlignmentType.LEFT, fill } = {}) =>
  new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: header ? { fill: NAVY, type: ShadingType.CLEAR, color: "auto" } : fill ? { fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: align, children: [run(String(text), { bold: header, color: header ? "FFFFFF" : undefined, size: 19 })] })],
  });
const table = (widths, header, rows, { alignRight = [] } = {}) =>
  new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], { header: true, align: alignRight.includes(i) ? AlignmentType.RIGHT : AlignmentType.LEFT })) }),
      ...rows.map((r, ri) =>
        new TableRow({
          children: r.map((v, i) => cell(v, widths[i], { align: alignRight.includes(i) ? AlignmentType.RIGHT : AlignmentType.LEFT, fill: ri % 2 ? "F3F6F9" : undefined })),
        })),
    ],
  });
const image = (file, widthPx, caption) => [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "jpg", data: fs.readFileSync(file), transformation: { width: widthPx, height: widthPx }, altText: { title: caption, description: caption, name: path.basename(file) } })],
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [run(caption, { italics: true, size: 18, color: "555555" })] }),
];
const spacer = () => new Paragraph({ children: [], spacing: { after: 60 } });

// ---------- content ----------
const resultsRows = IMAGES.map((n, i) => [
  `Photo ${i + 1}`,
  TRUE[n],
  `${yolo[n].predicted} (${signed(yolo[n].predicted - TRUE[n])})`,
  `${classical[n].predicted} (${signed(classical[n].predicted - TRUE[n])})`,
  `${yolo[n].ms.toFixed(0)} ms`,
]);

const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [run("Scaffold plank counter")] }),
  para([run("Test task report for Ready to Bloom · Elvis Kerebi · 4 November 2026", { color: "555555" })], { spacing: { after: 240 } }),

  h1("Summary"),
  rich([
    "I built a tool that counts the planks in a photo of a scaffold stack and draws a numbered box on each one. The five test photos show flat steel ",
    ["walking planks", { bold: true }],
    ", not round pipes, so the tool counts the plank ends you see at the front of each stack. The same method works for pipes once it is shown pipe examples.",
  ]),
  bullet([["Main method (a trained AI model): ", { bold: true }], `it counted ${accuracy(yolo)}% of all planks correctly, and was off by ${mae(yolo)} planks per photo on average. It was never tested on a photo it had learned from.`]),
  bullet([["Simple method (no AI training): ", { bold: true }], `${accuracy(classical)}% correct. Perfect on the neat stacks (photos 1 and 5), but it misses planks that stick out or sit back in the stack.`]),
  bullet([["On iPhone: ", { bold: true }], `the model converts to Apple's format (Core ML). It is ${sizeMb} MB and takes about ${medianMs} thousandths of a second per photo on an Apple M4 chip.`]),
  bullet([["Biggest limit: ", { bold: true }], "only five photos to learn from. The few remaining mistakes are on photo types it barely saw, like the view from above in photo 4. More real photos will fix this better than more tweaking."]),

  h1("1. How I set the right answers"),
  para("The photos came without correct counts, so I counted every plank by hand, zoomed in 3 times. As a check, a friend counted them separately and got the same totals: 54, 27, 76, 18 and 60. I counted with these rules, and every training example follows them:"),
  bullet([["One plank = one front end, ", { bold: true }], "including the shiny strip (lip) on top of it. The lip belongs to the plank; it is not a separate plank."]),
  bullet([["Every plank end has two holes. ", { bold: true }], "Two holes side by side confirm a plank."]),
  bullet([["Planks that stick out or sit back still count, ", { bold: true }], "even when their holes are hidden behind the plank next to them."]),
  bullet([["Planks cut off at the edge of the photo count ", { bold: true }], "if you can still tell they are a plank. Stacks in the background are not counted."]),

  h1("2. How it works"),
  h2("Step 1: the user marks the stack"),
  para("CountX already asks the user to mark the area to count. I use the same idea: the user marks each column of the stack with four corners. Everything outside is greyed out, so other stacks in the background are never counted. The model learns from these same marked areas, so training and real use match."),

  h2("Simple method (no training needed)"),
  numbered("Straighten each marked column so it looks square-on, like a photo taken from straight in front."),
  numbered([["Count the hole pairs: ", { bold: true }], "two dark dots side by side in a row = one plank."]),
  numbered([["Double-check with the gaps: ", { bold: true }], "count the thin dark lines between planks as a second count. If the two counts disagree, mark that column so the app can ask the user to check it."]),
  para("It needs no training photos and takes about 12 thousandths of a second. It is perfect on neat stacks but cannot see planks that stick out, because their holes are hidden. It correctly warned on every photo it got wrong."),

  h2("Main method: an AI model taught with only five photos"),
  bullet([["Start from a model that already knows how to see: ", { bold: true }], "YOLO11n, a small, fast object-detection model, already trained on everyday photos. It knows edges, corners and metal shine, so it only needs to learn what a plank looks like."]),
  bullet([["Making the examples: ", { bold: true }], "the simple method drew the first boxes on the neat stacks; I drew the boxes by hand on the messy ones (photos 2 and 4). Every photo's boxes were checked against the hand count, and the code refuses to continue if they don't match."]),
  bullet([["Stretching five photos into thousands: ", { bold: true }], "each photo is shown many times with small random changes: zoomed, slightly tilted, brighter or darker, mirrored left-to-right, or stitched with other photos. This teaches the model what stays the same. Photos are never turned upside down, because real stacks never are."]),
  bullet([["Tidying the answer: ", { bold: true }], "three simple rules clean up the model's boxes. Remove a half-plank box that sits inside a full one. Ignore boxes outside the marked columns. Join two halves of the same plank found side by side."]),
  bullet([["Fair testing: ", { bold: true }], "I trained five separate models. Each learned from four photos and was tested on the fifth, the one it had never seen. The final model for the app learns from all five."]),

  h1("3. Results"),
  para("Every count below comes from a model that had never seen that photo. Times are per photo on an Apple M4 MacBook."),
  table([1700, 1300, 2300, 2300, 1426], ["Photo", "Correct", "AI model (off by)", "Simple method (off by)", "AI time"], resultsRows, { alignRight: [1, 2, 3, 4] }),
  spacer(),
  table([3326, 2850, 2850], ["", "AI model", "Simple method"], [
    ["Planks counted correctly (all photos)", `${accuracy(yolo)}%`, `${accuracy(classical)}%`],
    ["Off by, on average per photo", `${mae(yolo)} planks`, `${mae(classical)} planks`],
    ["Time per photo on the Mac", `${meanMs(yolo)} ms`, `${meanMs(classical)} ms`],
    ["Time per photo in Apple's format (Core ML)", `${medianMs} ms`, "not needed"],
  ], { alignRight: [1, 2] }),
  para([run("ms = thousandths of a second.", { italics: true, size: 18, color: "555555" })]),
  h2("What else I tried, and why I stopped"),
  bullet("A bigger model with fewer random changes did worse (92.8%). With only four photos to learn from, it memorised them instead of learning what a plank is, and photo 4 dropped to 11."),
  bullet("Two ways of drawing the training boxes scored 96% and 98%. Training the same model twice can differ by that much on its own, so with five photos the gap is luck, not a real improvement."),
  bullet("If I kept changing settings until these five photos scored perfectly, the model would look great here but do worse on new photos. So I stopped and report the fair numbers."),

  h1("4. Where it goes wrong"),
  bullet([["View from above, thin planks pressed together (photo 4, 3 short): ", { bold: true }], "two thin planks sometimes get one box. It is the only photo taken from above, so the model had nothing like it to learn from."]),
  bullet([["Double boxes on one plank (photos 3 and 5, 2 extra): ", { bold: true }], "near the bottom of a column, where it is darker, the model sometimes draws two boxes on one plank, shifted by half a plank. They overlap too little for the tidy-up rules to catch."]),
  bullet([["Planks sticking out (simple method only, photos 2 and 4): ", { bold: true }], "the holes are hidden, so the hole counter cannot see these planks. Its double-check caught both photos and warned."]),
  bullet([["Glare, watermark, small photos: ", { bold: true }], "the top planks in photo 2 are washed out by light, photo 3's bottom row is under a watermark, and photos 1–3 are small (447 pixels wide), so some planks are only 12–20 pixels tall."]),

  h1("5. Putting it in the iOS app"),
  h2("The model file"),
  rich([`The final model converts to Apple's format, `, ["Core ML", { bold: true }], `: one ${sizeMb} MB file that runs on the iPhone's built-in AI chip (the Neural Engine). It takes about ${medianMs} ms per photo on an M4; recent iPhones have a similar chip, so it should be about as fast, though that still needs testing on a real phone. On these five photos the iPhone version matched the original model exactly or within 1 plank on four photos, and was 3 off on photo 4. The small gap comes from how it prepares the photo (a fixed square size) and how it filters overlapping boxes; these settings should be matched and checked on new photos before release.`]),
  h2("How it would run in CountX"),
  numbered("The user takes a photo and marks the stack columns (CountX's existing \"select range\" screen, with one shape per column).", 2),
  numbered("The app greys out everything outside and runs the model through Apple's Vision framework.", 2),
  numbered("The app applies the three tidy-up rules: about 40 lines of Swift, no extra model.", 2),
  numbered("The app shows numbered boxes and the total. The user taps to add or remove a plank, as CountX already does.", 2),
  h2("What it needs to work on real sites"),
  bullet([["More real photos: ", { bold: true }], "start with 50–100 stack photos from different sites, lighting and angles, including views from above and messy stacks like photos 2 and 4. Label them with the rules in section 1."]),
  bullet([["Learn from users' fixes: ", { bold: true }], "every time a user taps to add or remove a plank, that is a free correct answer. Save it (with permission) and use it to retrain the model regularly."]),
  bullet([["Use the full camera photo: ", { bold: true }], "count on the full-size photo (12 MP or more), not a small copy. Thin planks then become 60+ pixels tall instead of 15, and much easier to see."]),
  bullet([["A safety check: ", { bold: true }], "run the simple method next to the model. It takes a few thousandths of a second; when the two disagree a lot, ask the user to check that column."]),
  bullet([["Keep some photos only for testing: ", { bold: true }], "keep photos from sites the model never learned from, so every new version is tested fairly before release."]),

  h1("6. What is included"),
  bullet([["Marked photos: ", { bold: true }], "all five photos with every plank boxed, numbered and counted (in the appendix and in the images folder)."]),
  bullet([["Code: ", { bold: true }], "a Python project with tests. One command runs each part: plank-count classical, plank-count yolo, plank-count coreml. Setup steps are in the README. Your photos are not inside the code, since they are confidential."]),
  bullet([["iPhone model: ", { bold: true }], "the final model file in Apple's Core ML format."]),

  h1("Appendix: marked photos"),
  para("Each photo was counted by the model that had never seen it. Boxes are numbered top to bottom; the total is in the dark bar."),
  ...IMAGES.flatMap((n, i) =>
    image(path.join(OUT, "yolo", `${n}.jpg`), 320, `Photo ${i + 1}: counted ${yolo[n].predicted}, correct ${TRUE[n]}`)),
];

// ---------- document ----------
const doc = new Document({
  creator: "Elvis Kerebi",
  title: "Scaffold plank counter: test task report",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 44, bold: true, color: NAVY }, paragraph: { spacing: { before: 0, after: 80 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 28, bold: true, color: NAVY }, paragraph: { spacing: { before: 300, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 23, bold: true, color: "333333" }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [run("Scaffold plank counter · ", { size: 16, color: "888888" }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: "888888" })] })] }),
    },
    children,
  }],
});

const target = path.join(ROOT, "deliverables", "Scaffold_Plank_Counter_Report.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(target, buf);
  console.log(target);
});
