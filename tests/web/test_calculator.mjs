import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { ERROR_CODES, WARNING_CODES, assess } from "../../web/model.js";
import { LOCALES } from "../../web/i18n.js";

const card = JSON.parse(readFileSync(new URL("../../src/painpredict/assets/model_card.json", import.meta.url), "utf8"));

function codes(result, field) {
  return result.warnings.filter((item) => item.field === field).map((item) => item.code);
}

test("both locales contain every interface and validation message", () => {
  assert.deepEqual(Object.keys(LOCALES.en).sort(), Object.keys(LOCALES.zh).sort());
  for (const code of ERROR_CODES) {
    assert.ok(LOCALES.en[`error.${code}`]);
    assert.ok(LOCALES.zh[`error.${code}`]);
  }
  for (const code of WARNING_CODES) {
    assert.ok(LOCALES.en[`warning.${code}`]);
    assert.ok(LOCALES.zh[`warning.${code}`]);
  }
});

test("preprocedural NRS is required, integral and within 0 to 10", () => {
  assert.deepEqual(assess(card, {}).errors.map((item) => item.code), ["required"]);
  assert.equal(assess(card, { Pre_PainScore: "3.5", "D-dimer": "226", AFP: "5.23", Albumin: "40" }).errors[0].code, "not_integer");
  assert.equal(assess(card, { Pre_PainScore: "11", "D-dimer": "226", AFP: "5.23", Albumin: "40" }).errors[0].code, "out_of_range");
  assert.equal(assess(card, { Pre_PainScore: "1,5", "D-dimer": "226", AFP: "5.23", Albumin: "40" }).errors[0].code, "decimal_comma");
});

test("invalid laboratory entries block calculation without converting them", () => {
  const result = assess(card, { Pre_PainScore: "1", "D-dimer": "-5", AFP: "12 ng/mL", Albumin: "3,5" });
  assert.equal(result.ok, false);
  assert.equal(result.risk, null);
  assert.deepEqual(result.errors.map((item) => [item.field, item.code]), [
    ["D-dimer", "negative"],
    ["AFP", "not_numeric"],
    ["Albumin", "decimal_comma"],
  ]);
});

test("blank laboratory values use median imputation and unusual values warn", () => {
  const missing = assess(card, { Pre_PainScore: "2", "D-dimer": "", AFP: "  ", Albumin: "40" });
  assert.equal(missing.ok, true);
  assert.deepEqual(codes(missing, "D-dimer"), ["missing_imputed"]);
  assert.deepEqual(codes(missing, "AFP"), ["missing_imputed"]);
  assert.equal(missing.terms.find((row) => row.variable === "D-dimer").used, 226);
  const zero = assess(card, { Pre_PainScore: "1", "D-dimer": "0", AFP: "5.23", Albumin: "40" });
  assert.ok(codes(zero, "D-dimer").includes("implausible_zero"));

  const abnormal = assess(card, { Pre_PainScore: "8", "D-dimer": "2", AFP: "0.1", Albumin: "4.2" });
  assert.equal(abnormal.ok, true);
  assert.ok(codes(abnormal, "Pre_PainScore").includes("above_development_range"));
  assert.ok(codes(abnormal, "D-dimer").includes("suspected_unit_scale"));
  assert.ok(codes(abnormal, "AFP").includes("suspected_decimal"));
  assert.ok(codes(abnormal, "Albumin").includes("suspected_g_per_dl"));
  assert.equal(abnormal.terms.find((row) => row.variable === "Albumin").raw, 4.2);
});

test("values outside the capping bounds use the published bounds", () => {
  const high = assess(card, { Pre_PainScore: "1", "D-dimer": "50000", AFP: "2000000", Albumin: "70" });
  assert.equal(high.terms.find((row) => row.variable === "D-dimer").used, 962);
  assert.equal(high.terms.find((row) => row.variable === "AFP").used, 152.5725);
  assert.equal(high.terms.find((row) => row.variable === "Albumin").used, 55.35000000000001);
  assert.ok(codes(high, "D-dimer").includes("extreme_magnitude"));
  assert.ok(codes(high, "AFP").includes("extreme_magnitude"));

  const low = assess(card, { Pre_PainScore: "1", "D-dimer": "20", AFP: "2", Albumin: "20" });
  assert.equal(low.terms.find((row) => row.variable === "Albumin").used, 22.95);
  assert.ok(codes(low, "Albumin").includes("capped_low"));
  assert.ok(!codes(low, "Albumin").includes("suspected_g_per_dl"));
});

test("capping and the missing indicator follow the published equation", () => {
  const capped = assess(card, { Pre_PainScore: "1", "D-dimer": "5000", AFP: "5.23", Albumin: "40" });
  const bound = assess(card, { Pre_PainScore: "1", "D-dimer": "962", AFP: "5.23", Albumin: "40" });
  assert.equal(capped.risk, bound.risk);

  const absent = assess(card, { Pre_PainScore: "1", "D-dimer": "", AFP: "5.23", Albumin: "40" });
  const median = assess(card, { Pre_PainScore: "1", "D-dimer": "226", AFP: "5.23", Albumin: "40" });
  assert.notEqual(absent.risk, median.risk);
  assert.ok(absent.risk >= 0 && absent.risk <= 1);
});
