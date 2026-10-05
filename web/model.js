/**
 * Browser-side scoring for the published four-variable ridge logistic model.
 *
 * Coefficients, medians and capping bounds come only from model_card.json.
 * The arithmetic matches painpredict.formula_risk.
 */

export const THRESHOLD = 0.2;

/** Input-quality checks. They warn only; they never rescale a value. */
export const INPUT_CHECKS = Object.freeze({
  albuminGdLBelow: 8,
  ddimerImplausibleBelow: 10,
  afpImplausibleBelow: 0.5,
  ddimerExtremeAbove: 20000,
  afpExtremeAbove: 1e6,
  albuminExtremeAbove: 100,
});

export const ERROR_CODES = Object.freeze([
  "required",
  "out_of_range",
  "not_integer",
  "negative",
  "not_numeric",
  "decimal_comma",
  "not_finite",
]);

export const WARNING_CODES = Object.freeze([
  "missing_imputed",
  "below_development_range",
  "above_development_range",
  "capped_high",
  "capped_low",
  "suspected_g_per_dl",
  "suspected_unit_scale",
  "suspected_decimal",
  "extreme_magnitude",
  "implausible_zero",
]);

const NUMERIC = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i;

export function formatQuantity(value) {
  if (!Number.isFinite(value)) return "";
  const rounded = Math.round((value + Number.EPSILON) * 1e6) / 1e6;
  return String(rounded);
}

export function formatExact(value) {
  if (!Number.isFinite(value)) return "";
  const text = value.toPrecision(12);
  if (/e/i.test(text)) return text;
  return String(Number(text));
}

function parseNumeric(raw) {
  if (raw == null) return { missing: true };
  const text = String(raw).trim();
  if (text === "") return { missing: true };
  if (text.includes(",") || text.includes("，")) return { error: "decimal_comma" };
  if (!NUMERIC.test(text)) return { error: "not_numeric" };
  const value = Number(text);
  if (!Number.isFinite(value)) return { error: "not_finite" };
  return { value };
}

function isInteger(value) {
  return Math.abs(value - Math.round(value)) < 1e-9;
}

function clip(value, lower, upper) {
  let used = value;
  if (lower != null && used < lower) used = lower;
  if (upper != null && used > upper) used = upper;
  return used;
}

function termByName(card) {
  return new Map(card.equation.terms.map((term) => [term.variable, term]));
}

function qualityWarnings(variable, value) {
  const warnings = [];
  if (value === 0) warnings.push({ code: "implausible_zero" });
  if (variable === "Albumin" && value > 0 && value < INPUT_CHECKS.albuminGdLBelow) {
    warnings.push({ code: "suspected_g_per_dl" });
  }
  if (variable === "D-dimer" && value > 0 && value < INPUT_CHECKS.ddimerImplausibleBelow) {
    warnings.push({ code: "suspected_unit_scale" });
  }
  if (variable === "AFP" && value > 0 && value < INPUT_CHECKS.afpImplausibleBelow) {
    warnings.push({ code: "suspected_decimal" });
  }
  if (
    (variable === "D-dimer" && value > INPUT_CHECKS.ddimerExtremeAbove) ||
    (variable === "AFP" && value > INPUT_CHECKS.afpExtremeAbove) ||
    (variable === "Albumin" && value > INPUT_CHECKS.albuminExtremeAbove)
  ) {
    warnings.push({ code: "extreme_magnitude" });
  }
  return warnings;
}

/**
 * Validate inputs and, when every field is usable, compute the published risk.
 * Laboratory fields may be blank. Preprocedural NRS is required.
 */
export function assess(card, inputs) {
  const terms = termByName(card);
  const ranges = card.nomogram_ranges;
  const errors = [];
  const warnings = [];
  const parsed = new Map();

  for (const variable of card.predictors.map((item) => item.column)) {
    const result = parseNumeric(inputs?.[variable]);
    if (variable === "Pre_PainScore") {
      if (result.missing) {
        errors.push({ field: variable, code: "required" });
        continue;
      }
      if (result.error) {
        errors.push({ field: variable, code: result.error === "not_numeric" ? "not_numeric" : result.error });
        continue;
      }
      if (result.value < 0 || result.value > 10) {
        errors.push({ field: variable, code: "out_of_range" });
        continue;
      }
      if (!isInteger(result.value)) {
        errors.push({ field: variable, code: "not_integer" });
        continue;
      }
      parsed.set(variable, result.value);
      continue;
    }
    if (result.error) {
      errors.push({ field: variable, code: result.error });
      continue;
    }
    if (result.missing) {
      parsed.set(variable, null);
      continue;
    }
    if (result.value < 0) {
      errors.push({ field: variable, code: "negative" });
      continue;
    }
    parsed.set(variable, result.value);
  }

  if (errors.length) {
    return {
      ok: false,
      errors,
      warnings: [],
      risk: null,
      aboveThreshold: null,
      linearPredictor: null,
      terms: [],
    };
  }

  let linearPredictor = card.equation.intercept;
  const rows = [];
  for (const variable of card.predictors.map((item) => item.column)) {
    const term = terms.get(variable);
    const [low, high] = ranges[variable];
    const raw = parsed.get(variable);
    const missing = raw == null;
    if (missing) {
      warnings.push({
        field: variable,
        code: "missing_imputed",
        median: term.median,
        coefficient: term.missing_coefficient,
      });
    } else {
      if (raw < low) warnings.push({ field: variable, code: "below_development_range", low, high });
      if (raw > high) warnings.push({ field: variable, code: "above_development_range", low, high });
      warnings.push(...qualityWarnings(variable, raw).map((item) => ({ field: variable, ...item })));
    }
    const filled = clip(missing ? term.median : raw, term.cap_lower, term.cap_upper);
    if (!missing && term.cap_upper != null && raw > term.cap_upper) {
      warnings.push({ field: variable, code: "capped_high", used: filled });
    }
    if (!missing && term.cap_lower != null && raw < term.cap_lower) {
      warnings.push({ field: variable, code: "capped_low", used: filled });
    }
    const contribution = term.coefficient * filled + term.missing_coefficient * (missing ? 1 : 0);
    linearPredictor += contribution;
    rows.push({ variable, raw, missing, used: filled, contribution });
  }

  const risk = 1 / (1 + Math.exp(-linearPredictor));
  return {
    ok: true,
    errors: [],
    warnings,
    risk,
    aboveThreshold: risk >= THRESHOLD,
    linearPredictor,
    terms: rows,
  };
}
