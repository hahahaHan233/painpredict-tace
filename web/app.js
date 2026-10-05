import { assess, formatExact, formatQuantity } from "./model.js";
import { translate } from "./i18n.js";

const STORAGE_KEY = "painpredict-tace-locale";
const EXAMPLE = {
  Pre_PainScore: "0",
  "D-dimer": "120",
  AFP: "3.1",
  Albumin: "44",
};

const state = {
  locale: "en",
  card: null,
  touched: false,
  showExampleNote: false,
};

const form = document.querySelector("#calculator");
const resultNode = document.querySelector("#result");
const performanceNode = document.querySelector("#performance");
const exampleNote = document.querySelector("#example-note");

function initialLocale() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "en" || saved === "zh") return saved;
  } catch {
    /* Private browsing can block storage; fall through to the browser language. */
  }
  return (navigator.language || "en").toLowerCase().startsWith("zh") ? "zh" : "en";
}

function rememberLocale(locale) {
  try {
    localStorage.setItem(STORAGE_KEY, locale);
  } catch {
    /* The selection still applies for this page view. */
  }
}

function text(key, params) {
  return translate(state.locale, key, params);
}

function fieldName(variable) {
  return text(`field.${variable}.label`);
}

function readInputs() {
  return Object.fromEntries(
    [...form.elements].filter((element) => element.name).map((element) => [element.name, element.value]),
  );
}

function warningText(warning) {
  return text(`warning.${warning.code}`, {
    low: formatQuantity(warning.low),
    high: formatQuantity(warning.high),
    used: formatQuantity(warning.used),
    median: formatQuantity(warning.median),
    coefficient: formatExact(warning.coefficient),
  });
}

function applyLocale() {
  document.documentElement.lang = state.locale === "zh" ? "zh-Hans" : "en";
  document.title = text("app.title");
  document.querySelector(".languages").setAttribute("aria-label", text("app.language"));
  for (const button of document.querySelectorAll("[data-locale]")) {
    button.setAttribute("aria-pressed", String(button.dataset.locale === state.locale));
  }
  for (const element of document.querySelectorAll("[data-i18n]")) {
    element.textContent = text(element.dataset.i18n);
  }
  if (state.card) {
    for (const field of document.querySelectorAll(".field")) {
      const variable = field.dataset.field;
      const [low, high] = state.card.nomogram_ranges[variable];
      const hint = field.querySelector("[data-hint]");
      hint.textContent = variable === "Pre_PainScore"
        ? text("field.helpNrs", { low: formatQuantity(low), high: formatQuantity(high) })
        : text("field.helpLab", {
          low: formatQuantity(low),
          high: formatQuantity(high),
          unit: text(`field.${variable}.unit`),
        });
    }
    const auc = state.card.development.out_of_fold_performance.auc;
    performanceNode.textContent = text("performance.summary", {
      n: state.card.development.patients,
      events: state.card.development.events,
      auc: auc.estimate.toFixed(3),
      aucLow: auc.ci_low.toFixed(3),
      aucHigh: auc.ci_high.toFixed(3),
    });
  }
  render();
}

function render() {
  exampleNote.hidden = !state.showExampleNote;
  const inputs = readInputs();
  const idle = !state.touched && Object.values(inputs).every((value) => String(value).trim() === "");
  for (const field of document.querySelectorAll(".field")) {
    field.classList.remove("has-error", "has-warning");
    const feedback = field.querySelector("[data-feedback]");
    feedback.textContent = "";
    feedback.className = "feedback";
    field.querySelector("input").removeAttribute("aria-invalid");
  }
  resultNode.replaceChildren();
  if (!state.card || idle) {
    const paragraph = document.createElement("p");
    paragraph.textContent = text("result.idle");
    resultNode.append(paragraph);
    return;
  }

  const assessment = assess(state.card, inputs);
  for (const field of document.querySelectorAll(".field")) {
    const errors = assessment.errors.filter((item) => item.field === field.dataset.field);
    const warnings = assessment.ok ? assessment.warnings.filter((item) => item.field === field.dataset.field) : [];
    const feedback = field.querySelector("[data-feedback]");
    if (errors.length) {
      field.classList.add("has-error");
      field.querySelector("input").setAttribute("aria-invalid", "true");
      feedback.classList.add("error");
      feedback.textContent = errors.map((item) => text(`error.${item.code}`)).join(" ");
    } else if (warnings.length) {
      field.classList.add("has-warning");
      feedback.classList.add("warning");
      feedback.textContent = warnings.map(warningText).join(" ");
    }
  }

  if (!assessment.ok) {
    const box = document.createElement("div");
    box.className = "error-box";
    const heading = document.createElement("h3");
    heading.textContent = text("result.blocked");
    const list = document.createElement("ul");
    for (const error of assessment.errors) {
      const item = document.createElement("li");
      item.textContent = `${fieldName(error.field)}: ${text(`error.${error.code}`)}`;
      list.append(item);
    }
    box.append(heading, list);
    resultNode.append(box);
    return;
  }

  const label = document.createElement("p");
  label.textContent = text("result.probabilityLabel");
  const probability = document.createElement("p");
  probability.className = "probability";
  probability.append(`${(assessment.risk * 100).toFixed(1)}%`);
  const exact = document.createElement("span");
  exact.textContent = assessment.risk.toFixed(6);
  probability.append(exact);
  const outcome = document.createElement("p");
  outcome.textContent = text("result.outcome");
  const threshold = document.createElement("p");
  threshold.textContent = text(assessment.aboveThreshold ? "result.aboveThreshold" : "result.belowThreshold");
  resultNode.append(label, probability, outcome, threshold);

  if (assessment.warnings.length) {
    const box = document.createElement("div");
    box.className = "warning-box";
    const heading = document.createElement("h3");
    heading.textContent = text("result.warnings");
    const list = document.createElement("ul");
    for (const warning of assessment.warnings) {
      const item = document.createElement("li");
      item.textContent = `${fieldName(warning.field)}: ${warningText(warning)}`;
      list.append(item);
    }
    box.append(heading, list);
    resultNode.append(box);
  }

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = text("result.details");
  const table = document.createElement("table");
  const head = document.createElement("tr");
  for (const value of ["", `${text("result.entered")} → ${text("result.used")}`, text("result.contribution")]) {
    const cell = document.createElement("th");
    cell.textContent = value;
    head.append(cell);
  }
  const tableHead = document.createElement("thead");
  tableHead.append(head);
  const body = document.createElement("tbody");
  const intercept = document.createElement("tr");
  for (const value of [text("result.intercept"), "", formatExact(state.card.equation.intercept)]) {
    const cell = document.createElement("td");
    cell.textContent = value;
    intercept.append(cell);
  }
  body.append(intercept);
  for (const row of assessment.terms) {
    const line = document.createElement("tr");
    const entered = row.missing ? text("result.notMeasured") : formatQuantity(row.raw);
    for (const value of [fieldName(row.variable), `${entered} → ${formatQuantity(row.used)}`, formatExact(row.contribution)]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      line.append(cell);
    }
    body.append(line);
  }
  const total = document.createElement("tr");
  for (const value of [text("result.linearPredictor"), "", formatExact(assessment.linearPredictor)]) {
    const cell = document.createElement("td");
    cell.textContent = value;
    total.append(cell);
  }
  body.append(total);
  table.append(tableHead, body);
  details.append(summary, table);
  resultNode.append(details);
}

for (const button of document.querySelectorAll("[data-locale]")) {
  button.addEventListener("click", () => {
    state.locale = button.dataset.locale;
    rememberLocale(state.locale);
    applyLocale();
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  state.touched = true;
  render();
});

form.addEventListener("input", () => {
  state.touched = true;
  state.showExampleNote = false;
  render();
});

document.querySelector("#example").addEventListener("click", () => {
  for (const [name, value] of Object.entries(EXAMPLE)) form.elements[name].value = value;
  state.touched = true;
  state.showExampleNote = true;
  render();
});

document.querySelector("#clear").addEventListener("click", () => {
  form.reset();
  state.touched = false;
  state.showExampleNote = false;
  render();
});

state.locale = initialLocale();
applyLocale();

try {
  const response = await fetch(new URL("./model_card.json", import.meta.url));
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  state.card = await response.json();
  applyLocale();
} catch (error) {
  resultNode.replaceChildren();
  const box = document.createElement("div");
  box.className = "error-box";
  const message = document.createElement("p");
  message.textContent = text("error.load");
  box.append(message);
  resultNode.append(box);
  console.error(error);
}
