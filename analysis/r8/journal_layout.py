"""Lay out the journal draft from article.json and the R8 reconstructed tables."""
from pathlib import Path
import csv, json, os
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor
V=Path(__file__).resolve().parents[1];out=V/'manuscript'
read=lambda n: json.loads((V/'source'/n).read_text('utf8'))
article=read('article.json');refs=read('references.json');figs=read('captions.json')
def tab(name):
    with (V/'tables'/name).open(encoding='utf-8-sig') as s:
        return [r for r in csv.DictReader(s) if r['stratum']=='all']
rates=tab('rates.csv');paired=tab('paired_effects.csv')
num=lambda r,k:f"{float(r[k]):.2f}".replace('-','−')
def rng(r,lo,hi):return '['+num(r,lo)+', '+num(r,hi)+']'
names={'C0_no_instance':'去实例关联','C0_depth_only':'仅深度误差','C0_warp_only':'仅配准误差','C0':'完整 C0'}
tables={}
tables['T1']=dict(title='表 1　正式比较的全部尝试成功率、静态准入和动力学条件成功率',columns=['方法','成功/分派','成功率 %\n[95% Wilson 区间]','静态准入/分派\n（%）','成功/进入动力学\n（%）'],width=[2,2.6,4.55,3.9,3.95],rows=[
[r['method'],r['success']+'/240',num(r,'rate_pct')+'\n'+rng(r,'wilson_low_pct','wilson_high_pct'),r['static_admitted']+'/240\n('+num(r,'static_pass_all_pct')+')',r['success']+'/'+r['dynamics_entered']+'\n('+num(r,'conditional_success_pct')+')'] for r in rates if r['phase']=='R6'],note='分派、静态准入及动力学进入分别按意图、锁定候选后的完整路径检查和至少一个原生物理步返回计数。本表三组的静态准入数与动力学进入数相同。')
tables['T2']=dict(title='表 2　正式比较的配对差、成败不一致和候选分流',columns=['比较','差值 [95% 区间]\n（百分点）','前者/后者\n独有成功','精确 p','候选不同\n（%）'],width=[2.65,5.2,3.0,2.25,3.9],rows=[
[r['contrast'].replace('-',' − '),('+' if float(r['delta_pp'])>0 else '')+num(r,'delta_pp')+' '+rng(r,'CI_low_pp','CI_high_pp'),r['first_only']+' / '+r['second_only'],f"{float(r['exact_mcnemar_two_sided']):.4f}".rstrip('0'),r['different_candidates']+'/240 ('+num(r,'different_pct')+')'] for r in paired if r['phase']=='R6'],note='差值方向为前者减后者。C0 − A 为预设主要比较；其余比较的 p 值未经多重性调整。相同空选择也留在分流率的分母。')
tables['T3']=dict(title='表 3　三个探索性机制消融与同单元完整 C0 的配对结果',columns=['方法','成功/80\n（%）','差值 [95% 区间]\n（百分点）','消融/参照\n独有成功','选中不同\n（n/80）'],width=[3.2,2.5,5.7,3.1,2.5],rows=[['完整 C0','10 (12.50)','参照','—','—']]+[
[names[r['contrast'].rsplit('-',1)[0]],next(u['success']+' ('+num(u,'rate_pct')+')' for u in rates if u['phase']=='R7' and u['method']==r['contrast'].rsplit('-',1)[0]),('+' if float(r['delta_pp'])>0 else '')+num(r,'delta_pp')+' '+rng(r,'CI_low_pp','CI_high_pp'),r['first_only']+' / '+r['second_only'],r['different_candidates']+'/80'] for r in paired if r['phase']=='R7'],note='参照组沿用 R6 的 80 份原始结果。仅深度误差组所有观测配对差均为零，重抽样区间退化，未检验等效性。')
doc=Document();sec=doc.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.left_margin=sec.right_margin=Cm(2.0);sec.top_margin=Cm(1.9);sec.bottom_margin=Cm(1.85);sec.header_distance=Cm(.8);sec.footer_distance=Cm(.85)
core=doc.core_properties;core.title=article[0][0][1];core.subject='RGB-D 六自由度抓取的配对物理仿真评价';core.author='';core.keywords='RGB-D, paired grasping, uncertainty, physical simulation'
def font(t,size=10.5,bold=False,cn='宋体',west='Times New Roman',color='16252B',italic=False):
    t.font.name=west;t.font.size=Pt(size);t.font.bold=bold;t.font.italic=italic;t.font.color.rgb=RGBColor.from_string(color)
    f=t._element.get_or_add_rPr().rFonts
    f.set(qn('w:eastAsia'),cn);f.set(qn('w:cs'),west);return t
def layout(p,align=WD_ALIGN_PARAGRAPH.LEFT,line=15.8,before=0,after=4,first=0):
    f=p.paragraph_format;p.alignment=align;f.line_spacing=Pt(line);f.space_before=Pt(before);f.space_after=Pt(after);f.first_line_indent=Pt(first);f.widow_control=True;return p
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10.5);normal.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
for name,sz in [('Heading 1',15),('Heading 2',11.8)]:
    st=doc.styles[name];st.font.name='Arial';st.font.size=Pt(sz);st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'微软雅黑')
hdr=layout(sec.header.paragraphs[0],line=10,after=0);font(hdr.add_run('RGB-D 实例分割与六自由度机器人抓取'),8,cn='微软雅黑',west='Arial',color='526367')
fp=layout(sec.footer.paragraphs[0],align=WD_ALIGN_PARAGRAPH.CENTER,line=11,after=0)
fr=fp.add_run();field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');fr._r.addnext(field)
def mathrun(value):
    run=OxmlElement('m:r')
    text=OxmlElement('m:t');text.set(qn('xml:space'),'preserve');text.text=value;run.append(text)
    return run
def subscript(symbol,index):
    out=OxmlElement('m:sSub');base=OxmlElement('m:e');base.append(mathrun(symbol))
    sub=OxmlElement('m:sub');sub.append(mathrun(index));out.append(base);out.append(sub);return out
def equation(p,first):
    result=OxmlElement('m:oMath')
    elements=([mathrun('H = '),subscript('∑','g'),mathrun(' π(g) a(g) min['),subscript('b','L'),mathrun('(g), '),subscript('b','R'),mathrun('(g)]')]
              if first else [mathrun('S = s H [1 − clip(o / 0.01, 0, 1)]')])
    for item in elements:result.append(item)
    p._p.append(result);font(p.add_run('       (1)' if first else '       (2)'),10.3)
def fieldrow(p,k,t):
    if k in ['h','h2']:
        p.style='Heading 1' if k=='h' else 'Heading 2'
        layout(p,line=23 if k=='h' else 18,before=6 if k=='h' else 3,after=3)
        p.paragraph_format.keep_with_next=True
        font(p.add_run(t),15 if k=='h' else 11.8,True,'微软雅黑','Arial','153F49')
    elif k in ['title','title_en']:
        layout(p,WD_ALIGN_PARAGRAPH.CENTER,line=26 if k=='title' else 17,after=6 if k=='title' else 7)
        shown=t.replace('六自由度机器人抓取方法','\n六自由度机器人抓取方法') if k=='title' else t.replace(' with RGB-D','\nwith RGB-D')
        font(p.add_run(shown),19.0 if k=='title' else 13,True,'微软雅黑','Times New Roman','132D34')
    elif k=='eq':
        layout(p,WD_ALIGN_PARAGRAPH.CENTER,line=19,after=4);equation(p,t.startswith('H'))
    elif k in ['keys','keye']:
        layout(p,line=14,after=4);font(p.add_run(t),9.3,cn='黑体',color='374C53')
    elif k=='en':
        layout(p,WD_ALIGN_PARAGRAPH.JUSTIFY,line=14.0,after=5);font(p.add_run(t),10.5)
    else:
        layout(p,WD_ALIGN_PARAGRAPH.JUSTIFY,line=20.0,after=6,first=22.4);font(p.add_run(t),11.1)
def para(t,size=9,lead=13,pre=0,aft=3,bold=False,cn='宋体'):
    p=layout(doc.add_paragraph(),line=lead,before=pre,after=aft);font(p.add_run(t),size,bold,cn);return p
def border(cell,kind,width):
    pr=cell._tc.get_or_add_tcPr();bd=pr.find(qn('w:tcBorders'))
    if bd is None:bd=OxmlElement('w:tcBorders');pr.append(bd)
    el=OxmlElement('w:'+kind);el.set(qn('w:val'),'single');el.set(qn('w:sz'),str(width));el.set(qn('w:color'),'49636B');bd.append(el)
def maketable(t):
    p=para(t['title'],10.1,15,5,4,True,'微软雅黑');p.paragraph_format.keep_with_next=True
    table=doc.add_table(rows=0,cols=len(t['columns']));table.autofit=False;table.alignment=WD_TABLE_ALIGNMENT.CENTER
    for col,w in zip(table.columns,t['width']):col.width=Cm(w)
    for i,values in enumerate([t['columns']]+t['rows']):
        row=table.add_row();nu=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(nu)
        for j,(c,v,w) in enumerate(zip(row.cells,values,t['width'])):
            c.width=Cm(w);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p=layout(c.paragraphs[0],WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER,line=12.4,before=2,after=2);font(p.add_run(v),9.0,i==0,'微软雅黑' if i==0 else '宋体')
            if i==0:border(c,'top',10);border(c,'bottom',5)
            if i==len(t['rows']):border(c,'bottom',10)
    para('注　'+t['note'],8.5,12.8,3,5)
def figure(k):
    item=figs[k];p=layout(doc.add_paragraph(),WD_ALIGN_PARAGRAPH.CENTER,line=15,after=1)
    p.paragraph_format.line_spacing=1;p.paragraph_format.keep_with_next=True
    p.add_run().add_picture(str(V/'figures'/(item['file']+'.png')),width=Cm(16.7))
    p=para(item['title'],10,15,0,1,True,'微软雅黑');p.paragraph_format.keep_with_next=True
    para(item['text'],9.2,14.2,0,5)
def mdtable(t):
    arr=[t['title'],'','| '+' | '.join(s.replace('\n',' ') for s in t['columns'])+' |','| '+' | '.join(['---']*len(t['columns']))+' |']
    arr+=['| '+' | '.join(s.replace('\n',' ') for s in rr)+' |' for rr in t['rows']]
    return '\n'.join(arr+['','注　'+t['note'],''])
md=[]
for n,page in enumerate(article):
    for k,t in page:
        if k=='table':
            maketable(tables[t]);md.append(mdtable(tables[t]))
        elif k=='figure':
            figure(t);z=figs[t];md.extend(['!['+z['title']+'](../figures/'+z['file']+'.png)',z['title']+'。'+z['text']])
        elif k=='refs':
            for i,(citation,url,_) in enumerate(refs,1):
                para(f'[{i}] {citation}',9.4,13.2,0,4);md.append(f'[{i}] {citation} [来源]({url})')
        else:
            p=doc.add_paragraph()
            if n and (k,t)==tuple(page[0]):p.paragraph_format.page_break_before=True
            fieldrow(p,k,t)
            if k in ['title','h','h2']:md.append({'title':'# ','h':'## ','h2':'### '}[k]+t)
            else:md.append(t)
doc.save(out/'R8_中文论文初稿.docx')
(out/'R8_中文论文初稿.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
(V/'tables/journal_tables.md').write_text('# 正文统计表\n\n'+'\n\n'.join(mdtable(t) for t in tables.values())+'\n',encoding='utf8')
(V/'figures/captions.md').write_text('# 主要结果图图注\n\n'+'\n\n'.join(f"## {z['title']}\n\n{z['text']}" for z in sorted(figs.values(),key=lambda j:j['file']))+'\n',encoding='utf8')
