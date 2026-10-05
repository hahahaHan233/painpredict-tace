import { readFileSync } from "node:fs";

import { assess } from "../../web/model.js";

const card = JSON.parse(readFileSync(new URL("../../src/painpredict/assets/model_card.json", import.meta.url), "utf8"));
const cases = JSON.parse(readFileSync(0, "utf8"));
const risks = cases.map((row) => {
  const result = assess(card, row);
  if (!result.ok) throw new Error(`Case was rejected: ${JSON.stringify(result.errors)}`);
  return result.risk;
});
process.stdout.write(JSON.stringify(risks));
