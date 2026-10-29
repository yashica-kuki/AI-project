"""Writes report.csv and a self-contained interactive dashboard (report.html).
Open report.html by double-clicking it. No server, no internet, no Streamlit needed."""
import pandas as pd

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TechDebt-AI report</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#667;--card:#f5f6f8;--bd:#dfe2e6;--acc:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8e8e8;--mut:#9aa3ad;--card:#1d2026;--bd:#323741;--acc:#6ea8ff}}
*{box-sizing:border-box}
body{font-family:system-ui,"Segoe UI",Arial,sans-serif;margin:0;padding:24px;background:var(--bg);color:var(--fg)}
h1{margin:0 0 4px;font-size:22px}h2{font-size:15px;margin:0 0 10px}
.mut{color:var(--mut);font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.kpi{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px;margin-top:2px}
.ctrl{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:8px 0 16px;font-size:13px}
.ctrl input[type=text]{padding:6px 10px;border:1px solid var(--bd);border-radius:6px;background:var(--bg);color:var(--fg);width:220px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px;overflow:auto}
.clk{cursor:pointer}.clk:hover rect,.clk:hover circle{opacity:.75}
table{border-collapse:collapse;font-size:12.5px;width:100%}
th,td{border-bottom:1px solid var(--bd);padding:6px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal;word-break:break-all}
th{cursor:pointer;position:sticky;top:0;background:var(--card);user-select:none}
tr[data-p]{cursor:pointer}tr[data-p]:hover{background:rgba(120,140,200,.12)}
.tw{max-height:460px;overflow:auto}
.chip{display:inline-block;background:var(--bg);border:1px solid var(--bd);border-radius:999px;padding:3px 10px;margin:0 6px 6px 0;font-size:12px}
.sh{display:grid;grid-template-columns:200px 1fr 90px;gap:8px;align-items:center;font-size:12px;margin:3px 0}
.sh .ar{position:relative;height:14px}.sh .ar i{position:absolute;top:0;height:14px;border-radius:3px}
.sh .ar:before{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;border-left:1px solid var(--mut)}
li{margin:3px 0}
</style></head><body>
<h1>TechDebt-AI report</h1>
<div class="mut" id="sub"></div>
<div class="kpis" id="kpis"></div>
<div class="ctrl">
  <label>Search path <input type="text" id="q" placeholder="e.g. src/ or utils"></label>
  <label>Min score <input type="range" id="min" min="0" max="1" step="0.01" value="0"> <span id="minv">0</span></label>
  <label>Files in bar chart <input type="range" id="top" min="5" max="50" step="1"> <span id="topv"></span></label>
</div>
<div class="grid">
  <div class="card"><h2 id="bart"></h2><div id="bar"></div></div>
  <div class="card"><h2>Complexity vs. churn <span class="mut">(dot size = lines of code, colour = score)</span></h2><div id="sc"></div></div>
</div>
<div class="card" id="detail" style="margin-top:16px"></div>
<div class="card" style="margin-top:16px"><h2>All files <span class="mut" id="tinfo"></span></h2><div class="tw" id="tbl"></div></div>
<p class="mut"><b>priority</b> = risk x (1 - coverage) (or the debt score if no model was trained), so risky and untested files come first.
<b>risk</b> = model-predicted chance of a bug fix in the next window. <b>debt_score</b> = 0-100 heuristic.
<b>max_ccn</b> = most complex function. <b>churn</b> = lines added+deleted. <b>fan_in/out</b> = files depending on this / modules it imports.</p>
<script>
const DATA=__DATA__, SCORE="__SCORE__", TOP=__TOP__;
(function(){
const D=DATA,S=SCORE,$=id=>document.getElementById(id);
const has=k=>D.some(r=>r[k]!=null);
const esc=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt=v=>v==null?"\u2013":(Number.isInteger(v)?v:Number(v).toFixed(2));
const short=(s,n)=>s.length>n?"\u2026"+s.slice(-n):s;
const maxS=Math.max(0.0001,...D.map(r=>r[S]||0));
const col=v=>"hsl("+(120*(1-Math.min((v||0)/maxS,1)))+",65%,45%)";
const cols=["priority","risk","debt_score","coverage","max_ccn","nloc","fan_in","fan_out","churn","bug_fixes","n_authors"].filter(has);
const st={q:"",min:0,top:Math.min(TOP,50),sort:S,dir:-1,sel:null};
$("top").value=st.top;$("min").max=maxS.toFixed(2);
const rows=()=>D.filter(r=>r.path.toLowerCase().includes(st.q)&&(r[S]||0)>=st.min);

function kpis(f){
  const hi=f.filter(r=>(r[S]||0)>=(S==="debt_score"?75:0.5)).length;
  const cv=f.map(r=>r.coverage).filter(v=>v!=null);
  const k=[["Files shown",f.length+" / "+D.length],[S==="debt_score"?"High debt (>=75)":"High priority (>=0.5)",hi],
    ["Avg coverage",cv.length?(cv.reduce((a,b)=>a+b,0)/cv.length).toFixed(0)+"%":"n/a"],
    ["Bug-fix commits",f.reduce((a,r)=>a+(r.bug_fixes||0),0)]];
  $("kpis").innerHTML=k.map(x=>'<div class="kpi"><span class="mut">'+x[0]+'</span><b>'+x[1]+'</b></div>').join("");
}
function bar(f){
  const top=[...f].sort((a,b)=>(b[S]||0)-(a[S]||0)).slice(0,st.top);
  $("bart").textContent="Top "+top.length+" files by "+S;
  const W=640,rh=24,lab=260,H=top.length*rh+8;
  $("bar").innerHTML='<svg viewBox="0 0 '+W+' '+H+'" width="100%">'+top.map((r,i)=>{
    const w=Math.max((r[S]||0)/maxS*(W-lab-50),1);
    return '<g class="clk" data-p="'+esc(r.path)+'"><title>'+esc(r.path)+": "+fmt(r[S])+'</title>'+
    '<text x="'+(lab-6)+'" y="'+(i*rh+17)+'" text-anchor="end" font-size="12" fill="currentColor">'+esc(short(r.path,38))+'</text>'+
    '<rect x="'+lab+'" y="'+(i*rh+4)+'" width="'+w+'" height="17" rx="3" fill="'+col(r[S])+'"/>'+
    '<text x="'+(lab+w+5)+'" y="'+(i*rh+17)+'" font-size="11" fill="currentColor">'+fmt(r[S])+'</text></g>';}).join("")+'</svg>';
}
function scatter(f){
  const W=640,H=380,m={l:48,r:14,t:10,b:42};
  const xmax=Math.max(1,...f.map(r=>Math.log1p(r.churn||0))),ymax=Math.max(1,...f.map(r=>r.max_ccn||0));
  const X=v=>m.l+Math.log1p(v||0)/xmax*(W-m.l-m.r),Y=v=>H-m.b-(v||0)/ymax*(H-m.t-m.b);
  let s='<svg viewBox="0 0 '+W+' '+H+'" width="100%"><g font-size="11" fill="currentColor" stroke="none">';
  s+='<line x1="'+m.l+'" y1="'+(H-m.b)+'" x2="'+(W-m.r)+'" y2="'+(H-m.b)+'" stroke="currentColor" opacity=".4"/><line x1="'+m.l+'" y1="'+m.t+'" x2="'+m.l+'" y2="'+(H-m.b)+'" stroke="currentColor" opacity=".4"/>';
  [0,.5,1].forEach(t=>{const v=Math.round(ymax*t);s+='<text x="'+(m.l-6)+'" y="'+(Y(v)+4)+'" text-anchor="end">'+v+'</text>';});
  for(let t=0;t<=Math.expm1(xmax);t=t?t*10:1){s+='<text x="'+X(t)+'" y="'+(H-m.b+15)+'" text-anchor="middle">'+t+'</text>';if(!t)t=0.5;}
  s+='<text x="'+(W/2)+'" y="'+(H-6)+'" text-anchor="middle">churn (lines changed, log scale)</text>';
  s+='<text x="12" y="'+(H/2)+'" transform="rotate(-90 12 '+(H/2)+')" text-anchor="middle">max function complexity</text></g>';
  s+=f.map(r=>'<g class="clk" data-p="'+esc(r.path)+'"><title>'+esc(r.path)+" | "+S+" "+fmt(r[S])+" | churn "+fmt(r.churn)+" | ccn "+fmt(r.max_ccn)+'</title><circle cx="'+X(r.churn)+'" cy="'+Y(r.max_ccn)+'" r="'+(3+Math.min(Math.sqrt(r.nloc||0)/4,12))+'" fill="'+col(r[S])+'" fill-opacity=".7" stroke="currentColor" stroke-opacity=".3"/></g>').join("");
  $("sc").innerHTML=s+"</svg>";
}
function table(f){
  const sorted=[...f].sort((a,b)=>{const x=a[st.sort],y=b[st.sort];
    return (typeof x==="string"||typeof y==="string")?String(x).localeCompare(String(y))*st.dir:((x==null?-1:x)-(y==null?-1:y))*st.dir;});
  const lim=sorted.slice(0,300);
  $("tinfo").textContent="(click a header to sort, a row for details"+(sorted.length>300?"; showing first 300":"")+")";
  const th=["path"].concat(cols).map(k=>'<th data-k="'+k+'">'+k+(st.sort===k?(st.dir<0?" \u25BC":" \u25B2"):"")+'</th>').join("");
  $("tbl").innerHTML="<table><thead><tr>"+th+"</tr></thead><tbody>"+lim.map(r=>'<tr data-p="'+esc(r.path)+'"><td>'+esc(r.path)+"</td>"+
    cols.map(k=>'<td'+(k===S?' style="font-weight:600;color:'+col(r[k])+'"':"")+">"+fmt(r[k])+"</td>").join("")+"</tr>").join("")+"</tbody></table>";
}
function detail(){
  const el=$("detail");
  if(!st.sel){el.innerHTML='<h2>File detail</h2><p class="mut">Click a bar, dot or table row to see why a file is flagged and how to improve it.</p>';return;}
  const r=D.find(x=>x.path===st.sel);if(!r)return;
  let h='<h2>'+esc(r.path)+'</h2>'+cols.map(k=>'<span class="chip"><b>'+k+'</b> '+fmt(r[k])+'</span>').join("");
  h+='<h2 style="margin-top:10px">Suggestions</h2><ul>'+String(r.suggestions||"No major issues").split(" | ").map(s=>"<li>"+esc(s)+"</li>").join("")+"</ul>";
  const sk=Object.keys(r).filter(k=>k.indexOf("shap_")===0&&r[k]!=null).sort((a,b)=>Math.abs(r[b])-Math.abs(r[a])).slice(0,10);
  if(sk.length){
    const mx=Math.max(1e-9,...sk.map(k=>Math.abs(r[k])));
    h+='<h2 style="margin-top:10px">Why the model flags it <span class="mut">(SHAP: red pushes risk up, green pushes it down)</span></h2>';
    h+=sk.map(k=>{const v=r[k],w=Math.abs(v)/mx*50,f=k.slice(5);
      return '<div class="sh"><span>'+f+" = "+fmt(r[f])+'</span><div class="ar"><i style="'+(v>=0?"left:50%":"left:"+(50-w)+"%")+";width:"+w+"%;background:"+(v>=0?"#d9534f":"#3a9d5d")+'"></i></div><span>'+(v>=0?"+":"")+v.toFixed(3)+"</span></div>";}).join("");
  }
  el.innerHTML=h;
}
function render(){const f=rows();kpis(f);bar(f);scatter(f);table(f);detail();
  $("minv").textContent=st.min.toFixed(2);$("topv").textContent=st.top;}
document.addEventListener("click",e=>{
  const th=e.target.closest("th[data-k]");
  if(th){const k=th.dataset.k;st.dir=(st.sort===k)?-st.dir:-1;st.sort=k;render();return;}
  const t=e.target.closest("[data-p]");
  if(t){st.sel=t.dataset.p;detail();$("detail").scrollIntoView({behavior:"smooth",block:"nearest"});}
});
$("q").addEventListener("input",e=>{st.q=e.target.value.toLowerCase();render();});
$("min").addEventListener("input",e=>{st.min=parseFloat(e.target.value);render();});
$("top").addEventListener("input",e=>{st.top=parseInt(e.target.value);render();});
$("sub").textContent=D.length+" files analysed. Scores are relative to this repository.";
render();
})();
</script></body></html>
"""


def write_reports(df: pd.DataFrame, top: int = 25, csv_path="report.csv", html_path="report.html"):
    df.to_csv(csv_path)
    d = df.copy()
    d.index.name = "path"
    d = d.reset_index()
    for c in d.select_dtypes("number").columns:
        d[c] = d[c].round(3)
    if "priority" in d and d["priority"].notna().any():
        score = "priority"
    elif "risk" in d and d["risk"].notna().any():
        score = "risk"
    else:
        score = "debt_score"
    data = d.to_json(orient="records").replace("</", "<\\/")
    page = (TEMPLATE.replace("__DATA__", data).replace("__SCORE__", score).replace("__TOP__", str(int(top))))
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Wrote {csv_path} and {html_path}  (open report.html in your browser)")
