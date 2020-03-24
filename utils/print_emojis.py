from src.emoji2label import l2edata01 as l2e

with open('/utils/emojis.txt', 'w') as f:
    for e in l2e:
        e_unicode = l2e[e]
        line = str(e) + '\t' + '\t' + e_unicode.encode('utf-8')
        f.writelines(line)
        f.writelines('\n')
