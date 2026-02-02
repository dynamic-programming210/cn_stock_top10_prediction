"""
Chinese Stock Sector Classification
Based on stock code prefixes and industry classification

Chinese A-share sector classification follows CSRC (China Securities Regulatory Commission)
industry classification standard. This module provides sector mappings and utilities.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import re

# ============ Sector Definitions (Chinese and English) ============
SECTOR_NAMES = {
    'bank': ('银行', 'Banking'),
    'insurance': ('保险', 'Insurance'),
    'securities': ('证券', 'Securities'),
    'real_estate': ('房地产', 'Real Estate'),
    'infrastructure': ('基建', 'Infrastructure'),
    'steel': ('钢铁', 'Steel'),
    'coal': ('煤炭', 'Coal'),
    'oil_gas': ('石油天然气', 'Oil & Gas'),
    'power': ('电力', 'Power & Utilities'),
    'chemicals': ('化工', 'Chemicals'),
    'auto': ('汽车', 'Automobile'),
    'home_appliance': ('家电', 'Home Appliances'),
    'food_beverage': ('食品饮料', 'Food & Beverage'),
    'liquor': ('白酒', 'Liquor'),
    'pharma': ('医药', 'Pharmaceuticals'),
    'medical_device': ('医疗器械', 'Medical Devices'),
    'electronics': ('电子', 'Electronics'),
    'semiconductor': ('半导体', 'Semiconductor'),
    'software': ('软件', 'Software'),
    'telecom': ('通信', 'Telecom'),
    'media': ('传媒', 'Media'),
    'retail': ('零售', 'Retail'),
    'tourism': ('旅游', 'Tourism'),
    'airline': ('航空', 'Airlines'),
    'shipping': ('航运', 'Shipping'),
    'railway': ('铁路', 'Railway'),
    'port': ('港口', 'Ports'),
    'agriculture': ('农业', 'Agriculture'),
    'textile': ('纺织服装', 'Textile & Apparel'),
    'construction': ('建筑', 'Construction'),
    'building_material': ('建材', 'Building Materials'),
    'machinery': ('机械', 'Machinery'),
    'military': ('军工', 'Defense & Military'),
    'new_energy': ('新能源', 'New Energy'),
    'ev': ('新能源车', 'Electric Vehicles'),
    'photovoltaic': ('光伏', 'Photovoltaic'),
    'environmental': ('环保', 'Environmental'),
    'nonferrous': ('有色金属', 'Non-ferrous Metals'),
    'precious_metal': ('贵金属', 'Precious Metals'),
    'conglomerate': ('综合', 'Conglomerate'),
    'other': ('其他', 'Other'),
}

# ============ Stock Code to Sector Mapping (Based on known stocks) ============
# This is a curated list of major stocks and their sectors
STOCK_SECTOR_MAP = {
    # === 银行 Banking ===
    '600000': 'bank',  # 浦发银行
    '600015': 'bank',  # 华夏银行
    '600016': 'bank',  # 民生银行
    '600036': 'bank',  # 招商银行
    '601009': 'bank',  # 南京银行
    '601166': 'bank',  # 兴业银行
    '601169': 'bank',  # 北京银行
    '601229': 'bank',  # 上海银行
    '601288': 'bank',  # 农业银行
    '601328': 'bank',  # 交通银行
    '601398': 'bank',  # 工商银行
    '601818': 'bank',  # 光大银行
    '601939': 'bank',  # 建设银行
    '601988': 'bank',  # 中国银行
    '601998': 'bank',  # 中信银行
    '600919': 'bank',  # 江苏银行
    '600926': 'bank',  # 杭州银行
    '002142': 'bank',  # 宁波银行
    '002807': 'bank',  # 江阴银行
    '002839': 'bank',  # 张家港行
    '002936': 'bank',  # 郑州银行
    '002948': 'bank',  # 青岛银行
    '002958': 'bank',  # 青农商行
    '002966': 'bank',  # 苏州银行
    '000001': 'bank',  # 平安银行
    '002839': 'bank',  # 张家港行
    
    # === 保险 Insurance ===
    '601318': 'insurance',  # 中国平安
    '601601': 'insurance',  # 中国太保
    '601628': 'insurance',  # 中国人寿
    '601336': 'insurance',  # 新华保险
    '000627': 'insurance',  # 天茂集团
    '600291': 'insurance',  # 西水股份
    
    # === 证券 Securities ===
    '600030': 'securities',  # 中信证券
    '600837': 'securities',  # 海通证券
    '600958': 'securities',  # 东方证券
    '600999': 'securities',  # 招商证券
    '601066': 'securities',  # 中信建投
    '601099': 'securities',  # 太平洋
    '601162': 'securities',  # 天风证券
    '601198': 'securities',  # 东兴证券
    '601211': 'securities',  # 国泰君安
    '601236': 'securities',  # 红塔证券
    '601375': 'securities',  # 中原证券
    '601377': 'securities',  # 兴业证券
    '601456': 'securities',  # 国联证券
    '601555': 'securities',  # 东吴证券
    '601688': 'securities',  # 华泰证券
    '601696': 'securities',  # 中银证券
    '601788': 'securities',  # 光大证券
    '601878': 'securities',  # 浙商证券
    '601881': 'securities',  # 中国银河
    '601901': 'securities',  # 方正证券
    '601990': 'securities',  # 南京证券
    '601995': 'securities',  # 中金公司
    '000166': 'securities',  # 申万宏源
    '000712': 'securities',  # 锦龙股份
    '000728': 'securities',  # 国元证券
    '000750': 'securities',  # 国海证券
    '000776': 'securities',  # 广发证券
    '002500': 'securities',  # 山西证券
    '002673': 'securities',  # 西部证券
    '002736': 'securities',  # 国信证券
    '002797': 'securities',  # 第一创业
    '002926': 'securities',  # 华西证券
    '002939': 'securities',  # 长城证券
    '002945': 'securities',  # 华林证券
    
    # === 房地产 Real Estate ===
    '600048': 'real_estate',  # 保利发展
    '600325': 'real_estate',  # 华发股份
    '600383': 'real_estate',  # 金地集团
    '600606': 'real_estate',  # 绿地控股
    '600663': 'real_estate',  # 陆家嘴
    '600823': 'real_estate',  # 世茂股份
    '601155': 'real_estate',  # 新城控股
    '000002': 'real_estate',  # 万科A
    '000024': 'real_estate',  # 招商蛇口
    '000031': 'real_estate',  # 大悦城
    '000402': 'real_estate',  # 金融街
    '000656': 'real_estate',  # 金科股份
    '000671': 'real_estate',  # 阳光城
    '001979': 'real_estate',  # 招商蛇口
    '002146': 'real_estate',  # 荣盛发展
    
    # === 白酒 Liquor ===
    '600519': 'liquor',  # 贵州茅台
    '000858': 'liquor',  # 五粮液
    '000568': 'liquor',  # 泸州老窖
    '002304': 'liquor',  # 洋河股份
    '000596': 'liquor',  # 古井贡酒
    '600779': 'liquor',  # 水井坊
    '600559': 'liquor',  # 老白干酒
    '600809': 'liquor',  # 山西汾酒
    '600702': 'liquor',  # 舍得酒业
    '603369': 'liquor',  # 今世缘
    '603589': 'liquor',  # 口子窖
    '000799': 'liquor',  # 酒鬼酒
    '600199': 'liquor',  # 金种子酒
    '600197': 'liquor',  # 伊力特
    '000860': 'liquor',  # 顺鑫农业(牛栏山)
    '603198': 'liquor',  # 迎驾贡酒
    
    # === 食品饮料 Food & Beverage ===
    '600887': 'food_beverage',  # 伊利股份
    '000895': 'food_beverage',  # 双汇发展
    '002557': 'food_beverage',  # 洽洽食品
    '603288': 'food_beverage',  # 海天味业
    '600597': 'food_beverage',  # 光明乳业
    '600298': 'food_beverage',  # 安琪酵母
    '002311': 'food_beverage',  # 海大集团
    '600872': 'food_beverage',  # 中炬高新
    '002568': 'food_beverage',  # 百润股份
    '603027': 'food_beverage',  # 千禾味业
    '002847': 'food_beverage',  # 盐津铺子
    '603345': 'food_beverage',  # 安井食品
    '002216': 'food_beverage',  # 三全食品
    '600419': 'food_beverage',  # 天润乳业
    '600073': 'food_beverage',  # 上海梅林
    
    # === 医药 Pharmaceuticals ===
    '600276': 'pharma',  # 恒瑞医药
    '000538': 'pharma',  # 云南白药
    '600085': 'pharma',  # 同仁堂
    '600196': 'pharma',  # 复星医药
    '000423': 'pharma',  # 东阿阿胶
    '600332': 'pharma',  # 白云山
    '002001': 'pharma',  # 新和成
    '002007': 'pharma',  # 华兰生物
    '002252': 'pharma',  # 上海莱士
    '002422': 'pharma',  # 科伦药业
    '600436': 'pharma',  # 片仔癀
    '600566': 'pharma',  # 济川药业
    '600867': 'pharma',  # 通化东宝
    '002603': 'pharma',  # 以岭药业
    '300015': 'pharma',  # 爱尔眼科
    '300122': 'pharma',  # 智飞生物
    '300347': 'pharma',  # 泰格医药
    '300760': 'pharma',  # 迈瑞医疗
    '600763': 'pharma',  # 通策医疗
    
    # === 电子 Electronics ===
    '600183': 'electronics',  # 生益科技
    '600584': 'electronics',  # 长电科技
    '600703': 'electronics',  # 三安光电
    '601138': 'electronics',  # 工业富联
    '002049': 'electronics',  # 紫光国微
    '002371': 'electronics',  # 北方华创
    '002415': 'electronics',  # 海康威视
    '002475': 'electronics',  # 立讯精密
    '002600': 'electronics',  # 领益智造
    '002938': 'electronics',  # 鹏鼎控股
    '000725': 'electronics',  # 京东方A
    '000100': 'electronics',  # TCL科技
    '002241': 'electronics',  # 歌尔股份
    '002008': 'electronics',  # 大族激光
    
    # === 半导体 Semiconductor ===
    '600460': 'semiconductor',  # 士兰微
    '603986': 'semiconductor',  # 兆易创新
    '603501': 'semiconductor',  # 韦尔股份
    '688981': 'semiconductor',  # 中芯国际
    '002185': 'semiconductor',  # 华天科技
    '002156': 'semiconductor',  # 通富微电
    '002129': 'semiconductor',  # 中环股份
    '002049': 'semiconductor',  # 紫光国微
    '600206': 'semiconductor',  # 有研新材
    
    # === 新能源/光伏 New Energy/Photovoltaic ===
    '601012': 'new_energy',  # 隆基绿能
    '600438': 'photovoltaic',  # 通威股份
    '002459': 'photovoltaic',  # 晶澳科技
    '601877': 'new_energy',  # 正泰电器
    '002506': 'photovoltaic',  # 协鑫集成
    '600732': 'photovoltaic',  # 爱旭股份
    '002129': 'photovoltaic',  # 中环股份
    '600089': 'new_energy',  # 特变电工
    '002202': 'new_energy',  # 金风科技
    '601615': 'new_energy',  # 明阳智能
    
    # === 新能源车 Electric Vehicles ===
    '002594': 'ev',  # 比亚迪
    '600733': 'ev',  # 北汽蓝谷
    '601238': 'ev',  # 广汽集团
    '000625': 'ev',  # 长安汽车
    '600104': 'ev',  # 上汽集团
    '002074': 'ev',  # 国轩高科
    '300750': 'ev',  # 宁德时代
    '002460': 'ev',  # 赣锋锂业
    '002466': 'ev',  # 天齐锂业
    '300014': 'ev',  # 亿纬锂能
    '300124': 'ev',  # 汇川技术
    '002812': 'ev',  # 恩捷股份
    '603659': 'ev',  # 璞泰来
    
    # === 汽车 Automobile ===
    '600104': 'auto',  # 上汽集团
    '000625': 'auto',  # 长安汽车
    '601238': 'auto',  # 广汽集团
    '600166': 'auto',  # 福田汽车
    '000800': 'auto',  # 一汽解放
    '000550': 'auto',  # 江铃汽车
    '600418': 'auto',  # 江淮汽车
    '002920': 'auto',  # 德赛西威
    '600741': 'auto',  # 华域汽车
    '000338': 'auto',  # 潍柴动力
    '002048': 'auto',  # 宁波华翔
    '600660': 'auto',  # 福耀玻璃
    
    # === 家电 Home Appliances ===
    '000333': 'home_appliance',  # 美的集团
    '000651': 'home_appliance',  # 格力电器
    '600690': 'home_appliance',  # 海尔智家
    '002032': 'home_appliance',  # 苏泊尔
    '002508': 'home_appliance',  # 老板电器
    '002242': 'home_appliance',  # 九阳股份
    '000521': 'home_appliance',  # 美菱电器
    '600060': 'home_appliance',  # 海信视像
    '000404': 'home_appliance',  # 长虹华意
    '002035': 'home_appliance',  # 华帝股份
    
    # === 钢铁 Steel ===
    '600019': 'steel',  # 宝钢股份
    '000709': 'steel',  # 河钢股份
    '000898': 'steel',  # 鞍钢股份
    '600010': 'steel',  # 包钢股份
    '600022': 'steel',  # 山东钢铁
    '600808': 'steel',  # 马钢股份
    '600307': 'steel',  # 酒钢宏兴
    '000932': 'steel',  # 华菱钢铁
    '000959': 'steel',  # 首钢股份
    '000825': 'steel',  # 太钢不锈
    '002110': 'steel',  # 三钢闽光
    '600231': 'steel',  # 凌钢股份
    
    # === 煤炭 Coal ===
    '601088': 'coal',  # 中国神华
    '600188': 'coal',  # 兖矿能源
    '601898': 'coal',  # 中煤能源
    '601225': 'coal',  # 陕西煤业
    '600395': 'coal',  # 盘江股份
    '600348': 'coal',  # 华阳股份
    '600123': 'coal',  # 兰花科创
    '000933': 'coal',  # 神火股份
    '000937': 'coal',  # 冀中能源
    '600408': 'coal',  # 安泰集团
    '601699': 'coal',  # 潞安环能
    '600971': 'coal',  # 恒源煤电
    '600985': 'coal',  # 淮北矿业
    '002128': 'coal',  # 露天煤业
    
    # === 石油天然气 Oil & Gas ===
    '600028': 'oil_gas',  # 中国石化
    '601857': 'oil_gas',  # 中国石油
    '601808': 'oil_gas',  # 中海油服
    '600583': 'oil_gas',  # 海油工程
    '600871': 'oil_gas',  # 石化油服
    '002353': 'oil_gas',  # 杰瑞股份
    '002554': 'oil_gas',  # 惠博普
    '600339': 'oil_gas',  # 中油工程
    '002278': 'oil_gas',  # 神开股份
    
    # === 电力 Power & Utilities ===
    '600900': 'power',  # 长江电力
    '600025': 'power',  # 华能水电
    '600886': 'power',  # 国投电力
    '601991': 'power',  # 大唐发电
    '600027': 'power',  # 华电国际
    '601985': 'power',  # 中国核电
    '600795': 'power',  # 国电电力
    '600674': 'power',  # 川投能源
    '000883': 'power',  # 湖北能源
    '000027': 'power',  # 深圳能源
    '000600': 'power',  # 建投能源
    '000966': 'power',  # 长源电力
    '600744': 'power',  # 华银电力
    '000875': 'power',  # 吉电股份
    '000539': 'power',  # 粤电力A
    
    # === 化工 Chemicals ===
    '600309': 'chemicals',  # 万华化学
    '000792': 'chemicals',  # 盐湖股份
    '002601': 'chemicals',  # 龙佰集团
    '600426': 'chemicals',  # 华鲁恒升
    '000830': 'chemicals',  # 鲁西化工
    '600352': 'chemicals',  # 浙江龙盛
    '002064': 'chemicals',  # 华峰化学
    '600143': 'chemicals',  # 金发科技
    '002092': 'chemicals',  # 中泰化学
    '000301': 'chemicals',  # 东方盛虹
    '600141': 'chemicals',  # 兴发集团
    '000683': 'chemicals',  # 远兴能源
    
    # === 有色金属 Non-ferrous Metals ===
    '601899': 'nonferrous',  # 紫金矿业
    '000630': 'nonferrous',  # 铜陵有色
    '601600': 'nonferrous',  # 中国铝业
    '000060': 'nonferrous',  # 中金岭南
    '600362': 'nonferrous',  # 江西铜业
    '600219': 'nonferrous',  # 南山铝业
    '000878': 'nonferrous',  # 云南铜业
    '000426': 'nonferrous',  # 兴业矿业
    '600497': 'nonferrous',  # 驰宏锌锗
    '000831': 'nonferrous',  # 中国稀土
    '600111': 'nonferrous',  # 北方稀土
    '600259': 'nonferrous',  # 广晟有色
    '000758': 'nonferrous',  # 中色股份
    
    # === 军工 Defense & Military ===
    '600893': 'military',  # 航发动力
    '600760': 'military',  # 中航沈飞
    '000768': 'military',  # 中航西飞
    '600118': 'military',  # 中国卫星
    '600372': 'military',  # 中航电子
    '600150': 'military',  # 中国船舶
    '601989': 'military',  # 中国重工
    '000519': 'military',  # 中兵红箭
    '600862': 'military',  # 中航高科
    '002179': 'military',  # 中航光电
    '002414': 'military',  # 高德红外
    '002025': 'military',  # 航天电器
    '600765': 'military',  # 中航重机
    '600879': 'military',  # 航天电子
    '002013': 'military',  # 中航机电
    
    # === 通信 Telecom ===
    '600050': 'telecom',  # 中国联通
    '601728': 'telecom',  # 中国电信
    '600941': 'telecom',  # 中国移动
    '000063': 'telecom',  # 中兴通讯
    '002281': 'telecom',  # 光迅科技
    '600498': 'telecom',  # 烽火通信
    '002396': 'telecom',  # 星网锐捷
    '002194': 'telecom',  # 武汉凡谷
    '300502': 'telecom',  # 新易盛
    
    # === 软件 Software ===
    '600570': 'software',  # 恒生电子
    '600588': 'software',  # 用友网络
    '002410': 'software',  # 广联达
    '000977': 'software',  # 浪潮信息
    '002230': 'software',  # 科大讯飞
    '603019': 'software',  # 中科曙光
    '002405': 'software',  # 四维图新
    '300033': 'software',  # 同花顺
    '300059': 'software',  # 东方财富
    '002027': 'software',  # 分众传媒
    
    # === 传媒 Media ===
    '002027': 'media',  # 分众传媒
    '600637': 'media',  # 东方明珠
    '000681': 'media',  # 视觉中国
    '002354': 'media',  # 天神娱乐
    '002607': 'media',  # 中公教育
    '300413': 'media',  # 芒果超媒
    '300251': 'media',  # 光线传媒
    '002555': 'media',  # 三七互娱
    '002602': 'media',  # 世纪华通
    '002624': 'media',  # 完美世界
    
    # === 零售 Retail ===
    '600827': 'retail',  # 百联股份
    '600697': 'retail',  # 欧亚集团
    '600694': 'retail',  # 大商股份
    '600729': 'retail',  # 重庆百货
    '000417': 'retail',  # 合肥百货
    '002024': 'retail',  # 苏宁易购
    '002251': 'retail',  # 步步高
    '601933': 'retail',  # 永辉超市
    '002264': 'retail',  # 新华都
    
    # === 航空 Airlines ===
    '600115': 'airline',  # 中国东航
    '601111': 'airline',  # 中国国航
    '600029': 'airline',  # 南方航空
    '600221': 'airline',  # 海航控股
    '002928': 'airline',  # 华夏航空
    '600897': 'airline',  # 厦门空港
    
    # === 航运 Shipping ===
    '601866': 'shipping',  # 中远海发
    '601919': 'shipping',  # 中远海控
    '601872': 'shipping',  # 招商轮船
    '600026': 'shipping',  # 中远海能
    '000039': 'shipping',  # 中集集团
    '002245': 'shipping',  # 蔚蓝锂芯
    '600428': 'shipping',  # 中远海特
    
    # === 铁路 Railway ===
    '601006': 'railway',  # 大秦铁路
    '600125': 'railway',  # 铁龙物流
    '601333': 'railway',  # 广深铁路
    '601766': 'railway',  # 中国中车
    '600528': 'railway',  # 中铁工业
    '600169': 'railway',  # 太原重工
    
    # === 港口 Ports ===
    '600018': 'port',  # 上港集团
    '601018': 'port',  # 宁波港
    '600717': 'port',  # 天津港
    '000905': 'port',  # 厦门港务
    '000507': 'port',  # 珠海港
    '601000': 'port',  # 唐山港
    
    # === 建筑 Construction ===
    '601668': 'construction',  # 中国建筑
    '601186': 'construction',  # 中国铁建
    '601390': 'construction',  # 中国中铁
    '601800': 'construction',  # 中国交建
    '600170': 'construction',  # 上海建工
    '002941': 'construction',  # 新疆交建
    '600502': 'construction',  # 安徽建工
    '600820': 'construction',  # 隧道股份
    '002051': 'construction',  # 中工国际
    
    # === 基建 Infrastructure ===
    '601618': 'infrastructure',  # 中国中冶
    '600068': 'infrastructure',  # 葛洲坝
    '601669': 'infrastructure',  # 中国电建
    '600039': 'infrastructure',  # 四川路桥
    
    # === 建材 Building Materials ===
    '600585': 'building_material',  # 海螺水泥
    '000401': 'building_material',  # 冀东水泥
    '600801': 'building_material',  # 华新水泥
    '000877': 'building_material',  # 天山股份
    '002271': 'building_material',  # 东方雨虹
    '000786': 'building_material',  # 北新建材
    '601992': 'building_material',  # 金隅集团
    '002067': 'building_material',  # 景兴纸业
    '600176': 'building_material',  # 中国巨石
    
    # === 机械 Machinery ===
    '600031': 'machinery',  # 三一重工
    '000157': 'machinery',  # 中联重科
    '002097': 'machinery',  # 山河智能
    '600815': 'machinery',  # 厦工股份
    '600761': 'machinery',  # 安徽合力
    '000528': 'machinery',  # 柳工
    '000425': 'machinery',  # 徐工机械
    '600984': 'machinery',  # 建设机械
    '002270': 'machinery',  # 华明装备
    
    # === 农业 Agriculture ===
    '000998': 'agriculture',  # 隆平高科
    '600598': 'agriculture',  # 北大荒
    '002385': 'agriculture',  # 大北农
    '600354': 'agriculture',  # 敦煌种业
    '000713': 'agriculture',  # 丰乐种业
    '002041': 'agriculture',  # 登海种业
    '600127': 'agriculture',  # 金健米业
    '601118': 'agriculture',  # 海南橡胶
    '600438': 'agriculture',  # 通威股份
    '002714': 'agriculture',  # 牧原股份
    '002157': 'agriculture',  # 正邦科技
    '000876': 'agriculture',  # 新希望
    '002311': 'agriculture',  # 海大集团
    
    # === 旅游 Tourism ===
    '600138': 'tourism',  # 中青旅
    '000888': 'tourism',  # 峨眉山A
    '000978': 'tourism',  # 桂林旅游
    '600749': 'tourism',  # 西藏旅游
    '600054': 'tourism',  # 黄山旅游
    '002159': 'tourism',  # 三特索道
    '000428': 'tourism',  # 华天酒店
    '601007': 'tourism',  # 金陵饭店
    '000613': 'tourism',  # 大东海A
    '600258': 'tourism',  # 首旅酒店
    '002186': 'tourism',  # 全聚德
    
    # === 纺织服装 Textile & Apparel ===
    '600398': 'textile',  # 海澜之家
    '002029': 'textile',  # 七匹狼
    '002291': 'textile',  # 星期六
    '002269': 'textile',  # 美邦服饰
    '600400': 'textile',  # 红豆股份
    '002563': 'textile',  # 森马服饰
    '603877': 'textile',  # 太平鸟
    '002612': 'textile',  # 朗姿股份
    
    # === 环保 Environmental ===
    '000544': 'environmental',  # 中原环保
    '600008': 'environmental',  # 首创环保
    '601200': 'environmental',  # 上海环境
    '000826': 'environmental',  # 启迪环境
    '002310': 'environmental',  # 东方园林
    '300055': 'environmental',  # 万邦达
    '300070': 'environmental',  # 碧水源
    '300190': 'environmental',  # 维尔利
    '603126': 'environmental',  # 中材节能
}


def get_sector_by_name_keywords(name: str) -> Optional[str]:
    """
    Infer sector from stock name using keywords
    """
    name = name.lower() if name else ''
    
    # Keyword to sector mapping
    keyword_map = {
        'bank': ['银行', 'bank'],
        'insurance': ['保险', '人寿', 'insurance'],
        'securities': ['证券', '券商', 'securities'],
        'real_estate': ['地产', '置业', '置地', '房产'],
        'liquor': ['白酒', '酒业', '茅台', '五粮液', '老窖', '汾酒'],
        'food_beverage': ['食品', '乳业', '饮料', '调味', '酵母'],
        'pharma': ['医药', '制药', '药业', '生物', '医疗'],
        'electronics': ['电子', '光电', '半导体', '芯片', '科技'],
        'semiconductor': ['半导体', '芯片', '集成电路', '晶圆'],
        'new_energy': ['新能源', '风电', '光伏', '太阳能'],
        'ev': ['锂电', '电池', '新能源车', '充电'],
        'auto': ['汽车', '车业', '整车', '配件'],
        'home_appliance': ['电器', '家电', '空调', '冰箱'],
        'steel': ['钢铁', '钢股', '特钢', '不锈钢'],
        'coal': ['煤炭', '煤业', '矿业', '能源'],
        'oil_gas': ['石油', '石化', '油气', '天然气'],
        'power': ['电力', '发电', '水电', '核电', '能源'],
        'chemicals': ['化学', '化工', '材料'],
        'nonferrous': ['有色', '铜业', '铝业', '锌', '稀土'],
        'military': ['航空', '航天', '军工', '船舶', '兵器'],
        'telecom': ['通信', '通讯', '电信', '移动'],
        'software': ['软件', '信息', '网络', '科技', '数据'],
        'media': ['传媒', '娱乐', '游戏', '影视'],
        'retail': ['百货', '超市', '零售', '商业'],
        'airline': ['航空', '机场'],
        'shipping': ['航运', '海运', '物流'],
        'railway': ['铁路', '轨道', '中车'],
        'port': ['港口', '港务'],
        'construction': ['建筑', '建工', '建设'],
        'infrastructure': ['基建', '工程'],
        'building_material': ['水泥', '建材', '玻璃'],
        'machinery': ['机械', '重工', '设备'],
        'agriculture': ['农业', '种业', '养殖', '饲料'],
        'tourism': ['旅游', '酒店', '景区'],
        'textile': ['纺织', '服装', '服饰'],
        'environmental': ['环保', '环境', '水务', '节能'],
    }
    
    for sector, keywords in keyword_map.items():
        for kw in keywords:
            if kw in name:
                return sector
    
    return None


def get_stock_sector(symbol: str, name: str = None) -> str:
    """
    Get sector for a stock, using multiple methods:
    1. Direct mapping from STOCK_SECTOR_MAP
    2. Name-based keyword matching
    3. Default to 'other'
    """
    # Clean symbol (remove exchange suffix if present)
    code = symbol.split('.')[0] if '.' in symbol else symbol
    code = code.zfill(6)  # Ensure 6 digits
    
    # Method 1: Direct lookup
    if code in STOCK_SECTOR_MAP:
        return STOCK_SECTOR_MAP[code]
    
    # Method 2: Name-based inference
    if name:
        sector = get_sector_by_name_keywords(name)
        if sector:
            return sector
    
    # Method 3: Code prefix heuristics (less reliable but useful as fallback)
    # This is a rough heuristic based on Chinese stock code patterns
    prefix = code[:3]
    
    # Some code-based hints (not always accurate)
    if prefix in ['600', '601']:
        # Large caps on Shanghai, diverse
        pass
    
    return 'other'


def get_sector_name(sector_code: str, chinese: bool = True) -> str:
    """Get human-readable sector name"""
    if sector_code in SECTOR_NAMES:
        return SECTOR_NAMES[sector_code][0 if chinese else 1]
    return '其他' if chinese else 'Other'


def add_sector_to_dataframe(df: pd.DataFrame, 
                           symbol_col: str = 'symbol',
                           name_col: str = 'name') -> pd.DataFrame:
    """
    Add sector columns to a dataframe
    
    Args:
        df: DataFrame with stock symbols
        symbol_col: Column name containing stock symbols
        name_col: Column name containing stock names (optional)
    
    Returns:
        DataFrame with added 'sector', 'sector_cn', 'sector_en' columns
    """
    df = df.copy()
    
    # Get sector for each stock
    if name_col and name_col in df.columns:
        df['sector'] = df.apply(
            lambda row: get_stock_sector(str(row[symbol_col]), str(row.get(name_col, ''))),
            axis=1
        )
    else:
        df['sector'] = df[symbol_col].apply(lambda x: get_stock_sector(str(x)))
    
    # Add Chinese and English names
    df['sector_cn'] = df['sector'].apply(lambda x: get_sector_name(x, chinese=True))
    df['sector_en'] = df['sector'].apply(lambda x: get_sector_name(x, chinese=False))
    
    return df


def get_sector_stats(df: pd.DataFrame, sector_col: str = 'sector') -> pd.DataFrame:
    """Get statistics by sector"""
    stats = df.groupby(sector_col).agg({
        'symbol': 'count',
    }).rename(columns={'symbol': 'count'})
    
    stats['sector_cn'] = stats.index.map(lambda x: get_sector_name(x, chinese=True))
    stats = stats.sort_values('count', ascending=False)
    
    return stats


if __name__ == "__main__":
    # Test the sector classification
    test_stocks = [
        ('600000', '浦发银行'),
        ('600519', '贵州茅台'),
        ('000858', '五粮液'),
        ('002594', '比亚迪'),
        ('600031', '三一重工'),
        ('000333', '美的集团'),
        ('600276', '恒瑞医药'),
        ('601012', '隆基绿能'),
        ('000001', '平安银行'),
        ('600036', '招商银行'),
    ]
    
    print("Sector Classification Test:")
    print("-" * 60)
    for symbol, name in test_stocks:
        sector = get_stock_sector(symbol, name)
        sector_cn = get_sector_name(sector, chinese=True)
        sector_en = get_sector_name(sector, chinese=False)
        print(f"{symbol} {name:12} -> {sector_cn} ({sector_en})")
