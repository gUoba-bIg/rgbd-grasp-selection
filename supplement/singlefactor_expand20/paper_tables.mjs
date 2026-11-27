import fs from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {Workbook,SpreadsheetFile} from "@oai/artifact-tool";

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),"..");
const read=async relative=>JSON.parse(await fs.readFile(path.join(root,relative),"utf8"));
const execution=await read("statistics/execution240.json");
const comparisons=(await read("statistics/paired_effects.json")).groups;
const ranking=(await read("statistics/rank_stability.json")).summaries;
const outcomeNames=await read("statistics/outcome_dictionary.json");
const factors=["depth_only","shift_only","missing_only","block_only","all"];
const cn={all:"合并",depth_only:"深度高斯噪声",shift_only:"水平配准偏移",missing_only:"15% 随机缺深",block_only:"矩形遮挡"};
const methods=["Raw","A","C0"];
const wb=Workbook.create();
const exec=wb.worksheets.add("执行结果"), paired=wb.worksheets.add("配对效应"), rank=wb.worksheets.add("排序稳定性"),
      stops=wb.worksheets.add("失败去向"), raw=wb.worksheets.add("逐试次");
const col=i=>String.fromCharCode(65+i);
const cells=(s,letters,row)=>s.getRange(letters+row);
const core="#2f485d", shade="#e7eef3", ink="#172d3b";
const face="Microsoft YaHei";
function layout(sheet,title,annotation,headings,widths,n) {
    sheet.showGridLines=false;
    const last=col(headings.length-1);
    const all=sheet.getRange("A1:"+last+n);
    all.format.font={name:face,size:10,color:ink};
    all.format.rowHeight=18;
    sheet.getRange("A2").values=[[title]];
    sheet.getRange("A2").format.font={name:face,size:15,bold:true,color:ink};
    sheet.getRange("A2").format.rowHeight=26;
    sheet.getRange("A3").values=[[annotation]];
    sheet.getRange("A3").format.font={name:face,size:10,italic:true,color:"#445965"};
    const h=sheet.getRange("A5:"+last+"5");
    h.values=[headings];
    h.format={fill:core,font:{name:face,bold:true,color:"#ffffff",size:10},horizontalAlignment:"center",verticalAlignment:"center",rowHeight:24};
    for(let k=0;k<widths.length;k++)sheet.getRange(col(k)+"1:"+col(k)+n).format.columnWidthPx=widths[k];
}
function stripe(sheet,r,last,total) {
    if(total||r%2===0)sheet.getRange("A"+r+":"+last+r).format.fill=total?shade:"#f5f8fa";
    if(total)sheet.getRange("A"+r+":"+last+r).format.font={name:face,bold:true,color:ink,size:10};
}
function close(x,y,label) {
    if(y===null){if(x!==null&&x!==""&&x!=="—")throw Error(label+" expected missing, obtained "+x);return;}
    if(typeof x!=="number"||Math.abs(x-y)>1e-10)throw Error(label+" expected "+y+", obtained "+x);
}
layout(raw,"表 S5  方法试次","来源：已核验的分派、返回及原生回执。每行保留一个方法试次。0 / 1 表示否 / 是。",
       ["试次","来源","基础场景","干预","方法","首次终点","有候选","静态准入","进入动力学","抓取成功","物理步","抽样名次","锁定候选"],
       [105,78,185,142,70,215,76,83,95,83,86,80,340],245);
raw.getRange("A6:M245").values=execution.records.map(r=>[r.trial_id,r.origin==="carried"?"沿用":"新增",r.scene,cn[r.condition],r.method,outcomeNames[r.failure_class],
       Number(r.selected),Number(r.static_admitted),Number(r.physics),Number(r.success),r.physical_steps,r.hash_rank,r.candidate]);
raw.freezePanes.freezeRows(5);
const checked=[];
function range(c){return "'逐试次'!$"+c+"$6:$"+c+"$245";}
function conditionalCount(r,cond,measure,reason) {
    const terms=[range("E"),"$B"+r];
    if(cond!=="all")terms.push(range("D"),"$A"+r);
    if(reason)terms.push(range("F"),'"'+reason+'"');
    if(measure)return "=SUMIFS("+[range(measure),...terms].join(",")+")";
    return "=COUNTIFS("+terms.join(",")+")";
}
layout(exec,"表 S1  单因素干预与抓取结果","分母为全部分派试次。区间为基础场景聚类重采样 20,000 次的百分位数。条件成功率仅以进入动力学的试次为分母。",
       ["干预","方法","全部试次","成功","成功率","95% 下限","95% 上限","静态准入","准入率","动力学次数","条件成功率","95% 下限","95% 上限"],
       [145,73,85,65,85,90,90,88,85,98,105,90,90],20);
let row=6;
for(const c of factors)for(const m of methods){
    const v=execution.groups.find(x=>x.condition===c&&x.method===m);
    const r=row++;const gc=v.grasp_success.ci95??[null,null],dc=v.conditional_success.ci95??[null,null];
    exec.getRange("A"+r+":M"+r).values=[[cn[c],m,null,null,null,...gc,null,null,null,null,...dc]];
    for(const [k,content] of Object.entries({C:conditionalCount(r,c),D:conditionalCount(r,c,"J"),E:"=D"+r+"/C"+r,
        H:conditionalCount(r,c,"H"),I:"=H"+r+"/C"+r,J:conditionalCount(r,c,"I"),K:'=IF(J'+r+'=0,"—",D'+r+'/J'+r+')'}))cells(exec,k,r).formulas=[[content]];
    stripe(exec,r,"M",c==="all");
    checked.push([exec,"C"+r,v.n],[exec,"D"+r,v.success],[exec,"E"+r,v.grasp_success.value],[exec,"H"+r,v.static_admitted],
                 [exec,"J"+r,v.entered_dynamics],[exec,"K"+r,v.conditional_success.value]);
}
for(const block of ["E6:G20","I6:I20","K6:M20"])exec.getRange(block).format.numberFormat="0.00%";
layout(paired,"表 S2  配对成功率差","差值为前者减后者。独有失败与另一方法的独有成功相对应。相同基础场景的观测及方法共同重采样。",
       ["干预","前者","后者","配对数","基础场景","前者独有成功","前者独有失败","共同成功","共同失败","锁定候选不同","差值","95% 下限","95% 上限"],
       [145,63,63,72,90,115,115,90,90,118,85,90,90],20);
row=6;
for(const c of factors)for(const [a,b] of [["A","Raw"],["C0","Raw"],["C0","A"]]){
    const q=comparisons.find(x=>x.condition===c&&x.first===a&&x.second===b);const r=row++;
    paired.getRange("A"+r+":M"+r).values=[[cn[c],a,b,q.n,q.scene_clusters,q.first_only_success,q.first_only_failure,q.both_success,q.both_failure,q.different_candidates,null,...q.difference.ci95]];
    cells(paired,"K",r).formulas=[["=(F"+r+"-G"+r+")/D"+r]];stripe(paired,r,"M",c==="all");checked.push([paired,"K"+r,q.difference.value]);
}
paired.getRange("K6:M20").format.numberFormat="0.0%;-0.0%;0.0%";
layout(rank,"表 S3  与同场景 clean 的排序稳定性","主分析为每种干预前 20 个执行单元。全部观测每种 120 个。无可比较顺序时 τ-b 留空，有效数另列。",
       ["分析集","干预","方法","观测数","Top-1 变化数","Top-1 变化率","Top-3 Jaccard","Jaccard 有效数","Kendall τ-b","τ-b 有效数","τ-b 95% 下限","τ-b 95% 上限"],
       [103,145,68,75,112,111,106,123,106,100,117,117],35);
row=6;
for(const pop of ["execution_sample","full_120_per_factor"])for(const c of factors)for(const m of methods){
    const v=ranking.find(a=>a.population===pop&&a.condition===c&&a.method===m);const r=row++;const t=v.kendall_tau_b,j=v.top3_jaccard;
    rank.getRange("A"+r+":L"+r).values=[[pop==="execution_sample"?"执行样本":"全部观测",cn[c],m,v.units,v.pre_E_top1_change.numerator,null,j.value,j.denominator,t.value,t.denominator,...(t.ci95??[null,null])]];
    cells(rank,"F",r).formulas=[["=E"+r+"/D"+r]];stripe(rank,r,"L",c==="all");checked.push([rank,"F"+r,v.pre_E_top1_change.value]);
}
rank.getRange("F6:F35").format.numberFormat="0.00%";
for(const k of ["G","I","K","L"])rank.getRange(k+"6:"+k+"35").format.numberFormat="0.000";
rank.freezePanes.freezeRows(5);
const codes=Object.keys(outcomeNames);
layout(stops,"表 S4  互斥终点与失败去向","每个方法试次只计入一个首次终点。动力学禁止碰撞与静态路径拒绝分别计数。",
        ["干预","方法","试次",...Object.values(outcomeNames)],[145,70,80,...codes.map(z=>z==="empty_old_E"?182:z==="empty_pool"?135:Math.max(78,outcomeNames[z].length*16))],20);
row=6;
for(const c of factors)for(const m of methods){
    const r=row++;const g=execution.groups.find(x=>x.condition===c&&x.method===m);
    stops.getRange("A"+r+":B"+r).values=[[cn[c],m]];cells(stops,"C",r).formulas=[[conditionalCount(r,c)]];
    for(let i=0;i<codes.length;i++){
        const cell=col(i+3);cells(stops,cell,r).formulas=[[conditionalCount(r,c,null,outcomeNames[codes[i]])]];
        checked.push([stops,cell+r,g.outcomes[codes[i]]]);
    }
    stripe(stops,r,col(codes.length+2),c==="all");
}
wb.recalculate();
for(const [sheet,cell,expected] of checked)close(sheet.getRange(cell).values[0][0],expected,sheet.name+"!"+cell);
const auditDir=path.join(root,"admission","visual_qa");
await fs.mkdir(auditDir,{recursive:true});
const inspect=await wb.inspect({kind:"table",range:"执行结果!A18:M20",include:"values,formulas",tableMaxRows:3,tableMaxCols:13,maxChars:4000});
const errors=await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:100},summary:"Scan for formula errors"});
for(const [s,area,id] of [[exec,"A1:M20","execution"],[paired,"A1:M20","paired"],[rank,"A1:L11","ranking"],[stops,"A1:O10","endpoints"],[raw,"A1:M10","trials"]]){
    const blob=await wb.render({sheetName:s.name,range:area,scale:1,format:"png"});
    await fs.writeFile(path.join(auditDir,id+".png"),new Uint8Array(await blob.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(root,"statistics","V1单因素扩充论文表格.xlsx"));
await fs.writeFile(path.join(auditDir,"workbook_qa.json"),JSON.stringify({compared_cells:checked.length,counts_match:true,
   terminal_summary:inspect.ndjson,error_scan:errors.ndjson,sheets:5},null,2));
