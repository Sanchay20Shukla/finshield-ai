// Faithful inference from exported XGBoost trees. Float32 matches DMatrix inputs.
export function features(t) {
  const outgoing = t.type !== 'CASH_IN';
  const full = outgoing && t.newbalanceOrig === 0 && Math.abs(t.oldbalanceOrg-t.amount) <= .01+1e-5*Math.abs(t.amount);
  return {
    transaction_type:t.type, amount:t.amount, origin_old_balance:t.oldbalanceOrg,
    origin_new_balance:t.newbalanceOrig, destination_old_balance:t.oldbalanceDest,
    destination_new_balance:t.newbalanceDest, origin_balance_delta:t.oldbalanceOrg-t.newbalanceOrig,
    destination_balance_delta:t.newbalanceDest-t.oldbalanceDest,
    amount_to_origin_balance_ratio:t.amount/(t.oldbalanceOrg+1),
    amount_to_destination_balance_ratio:t.amount/(t.oldbalanceDest+1),
    origin_balance_error:Math.abs(t.newbalanceOrig-(t.oldbalanceOrg+(outgoing?-t.amount:t.amount))),
    destination_balance_error:t.type==='PAYMENT' && (t.nameDest||'').startsWith('M')?0:Math.abs(t.newbalanceDest-(t.oldbalanceDest+t.amount)),
    zero_balance_after_transaction:+(t.newbalanceOrig===0), full_balance_transfer:+full,
    large_transaction:+(t.amount>=200000), transaction_velocity:t.transaction_velocity??1,
    account_transaction_count:t.account_transaction_count??0, average_account_amount:t.average_account_amount??0,
    deviation_from_account_average:t.account_transaction_count && t.average_account_amount?t.amount-t.average_account_amount:0,
    destination_frequency:t.destination_frequency??0, high_risk_transaction_type:+(['TRANSFER','CASH_OUT'].includes(t.type))
  };
}
export function score(model,t) {
  if (!['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER'].includes(t.type)) throw new Error('Unsupported transaction type');
  for (const key of ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest'])
    if (!Number.isFinite(t[key]) || t[key]<0 || t[key]>1e12) throw new Error('Amounts must be finite and between 0 and 1 trillion.');
  const f=features(t);
  const x=model.featureNames.map(name=>Math.fround(name.startsWith('type__transaction_type_')?+(t.type===name.slice(23)):f[name.split('__')[1]]));
  if(x.some(v=>!Number.isFinite(v))) throw new Error('Invalid model features');
  let margin=model.baseMargin;
  for(let tree of model.trees){while(Array.isArray(tree)) tree=x[tree[0]]<Math.fround(tree[1])?tree[2]:tree[3];margin=Math.fround(margin+Math.fround(tree));}
  const probability=1/(1+Math.exp(-margin));
  const rules=[];
  const add=(name,points,reason)=>rules.push({name,points,reason});
  if(t.type==='TRANSFER'&&t.amount>=200000) add('Large transfer',20,'Amount meets the 200,000 policy threshold.');
  if(t.type!=='CASH_IN'&&t.newbalanceOrig<=.01&&t.oldbalanceOrg>.01) add('Origin balance depleted',15,'The funded origin is left with no balance.');
  if(f.full_balance_transfer||(t.type!=='CASH_IN'&&t.oldbalanceOrg>.01&&Math.abs(t.amount-t.oldbalanceOrg)<=Math.max(.01,t.oldbalanceOrg*1e-5)&&t.newbalanceOrig<=.01)) add('Full balance transfer',25,'Amount approximately equals the available origin balance.');
  if(f.transaction_velocity>=4) add('High transaction velocity',20,'Four or more transactions within the window.');
  if(f.high_risk_transaction_type&&f.destination_frequency>=3) add('Repeated destination',15,'Destination appeared at least three times previously.');
  const multiple=f.average_account_amount>0?t.amount/f.average_account_amount:0;
  if(f.account_transaction_count>=3&&multiple>=3) add('Unusual account amount',15,'At least three times the historical average.');
  let behavior=0;
  if(f.transaction_velocity>1) behavior+=Math.min(25,(f.transaction_velocity-1)*6.25);
  if(f.account_transaction_count>=3&&multiple>1) behavior+=Math.min(25,(multiple-1)*6.25);
  if(f.destination_frequency>0) behavior+=Math.min(15,f.destination_frequency*3);
  if(f.zero_balance_after_transaction||f.full_balance_transfer) behavior+=f.full_balance_transfer?20:12;
  const error=(f.origin_balance_error+f.destination_balance_error)/Math.max(t.amount,1);
  if(error>.01) behavior+=Math.min(15,error*15);
  behavior=Math.round(Math.min(100,behavior)*1e4)/1e4;
  const ruleScore=Math.min(100,rules.reduce((a,r)=>a+r.points,0));
  const risk=Math.round((probability*65+ruleScore*.2+behavior*.15)*1e4)/1e4;
  return {probability,risk,decision:risk<=30?'APPROVE':risk<=60?'MONITOR':risk<=80?'MANUAL_REVIEW':'BLOCK',rules,ruleScore,behavior,features:f};
}
