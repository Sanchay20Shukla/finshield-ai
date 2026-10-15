import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {score} from '../web/engine.mjs';
const model=JSON.parse(readFileSync(new URL('../web/data/model.json',import.meta.url)));
const cases=JSON.parse(readFileSync(new URL('../web/data/cases.json',import.meta.url)));
test('browser XGBoost and policy outputs match Python on every exported case',()=>{
  assert.ok(cases.length>100);
  for(const c of cases){
    const result=score(model,c.transaction);
    assert.ok(Math.abs(result.probability-c.probability)<1e-6,`${c.id} probability`);
    assert.ok(Math.abs(result.risk-c.decision.final_risk_score)<.0002,`${c.id} risk`);
    assert.equal(result.decision,c.decision.decision,`${c.id} decision`);
  }
});
test('invalid custom inputs cannot silently produce a score',()=>{
  for(const amount of [-1,NaN,Infinity,1e13])assert.throws(()=>score(model,{...cases[0].transaction,amount}));
  assert.throws(()=>score(model,{...cases[0].transaction,type:'UNKNOWN'}));
});
