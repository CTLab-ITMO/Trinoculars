import spacy
from collections import Counter
import math
import numpy as np

nlp = spacy.load("ru_core_news_lg")

def analyze_text(text):
    doc = nlp(text)

    # === БАЗОВЫЙ АНАЛИЗ ТОКЕНОВ ===
    tokens = [token.text for token in doc]
    words = [token.text for token in doc if token.is_alpha]
    unique_words = set(words)
    stop_words = [token.text for token in doc if token.is_stop]
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

    # === МОРФОЛОГИЧЕСКИЙ АНАЛИЗ ===
    pos_counts = Counter([token.pos_ for token in doc if token.is_alpha])
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    unique_lemmas = set(lemmas)
    
    # === СИНТАКСИЧЕСКИЙ АНАЛИЗ ===
    dependencies = Counter([token.dep_ for token in doc if token.dep_ != ""])
    
    has_noun_chunks = False
    try:
        next(doc.noun_chunks, None)
        has_noun_chunks = True
    except NotImplementedError:
        pass
    
    # === ИМЕНОВАННЫЕ СУЩНОСТИ (NER) ===
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    entity_counts = Counter([ent.label_ for ent in doc.ents])
    
    # === ЛЕКСИЧЕСКОЕ РАЗНООБРАЗИЕ ===
    ttr = len(unique_words) / len(words) if words else 0
    
    def calculate_simplified_mtld(text_tokens, ttr_threshold=0.72):
        if len(text_tokens) < 10:
            return 0
        
        segments = []
        current_segment = []
        
        for token in text_tokens:
            current_segment.append(token)
            current_ttr = len(set(current_segment)) / len(current_segment)
            
            if current_ttr <= ttr_threshold and len(current_segment) >= 10:
                segments.append(current_segment)
                current_segment = []
        
        if current_segment:
            segments.append(current_segment)
            
        if not segments:
            return 0
        
        return len(text_tokens) / len(segments)
    
    mtld = calculate_simplified_mtld(words)
    
    # === ЧИТАБЕЛЬНОСТЬ ТЕКСТА ===
    sentences = list(doc.sents)
    sentence_lengths = [len(sent) for sent in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentences) if sentences else 0
    
    words_per_sentence = len(words) / len(sentences) if sentences else 0
    
    def count_syllables_ru(word):
        return len([c for c in word.lower() if c in 'аеёиоуыэюя'])
    
    syllables = sum(count_syllables_ru(word) for word in words)
    syllables_per_word = syllables / len(words) if words else 0
    flesh_kincaid = 206.835 - 1.3 * words_per_sentence - 60.1 * syllables_per_word
    
    # === СЛОЖНОСТЬ ТЕКСТА ===
    long_words = [word for word in words if count_syllables_ru(word) > 4]
    long_words_percent = len(long_words) / len(words) * 100 if words else 0
    
    # === ЧАСТОТНОСТЬ СЛОВ ===
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(10)
    
    # === ВЕКТОРНЫЕ ХАРАКТЕРИСТИКИ ===
    doc_vector = doc.vector
    vector_norm = np.linalg.norm(doc_vector)
    
    # === ПРЕДЛОЖЕНИЯ И ИХ ХАРАКТЕРИСТИКИ ===
    sentence_count = len(sentences)
    question_count = sum(1 for sent in sentences if sent.text.strip().endswith('?'))
    exclamation_count = sum(1 for sent in sentences if sent.text.strip().endswith('!'))
    
    # === ВЫВОД СТАТИСТИКИ ===
    print(f"\n📊 ПОЛНЫЙ АНАЛИЗ ТЕКСТА")
    print(f"\n=== БАЗОВАЯ СТАТИСТИКА ===")
    print(f"- Всего токенов: {len(tokens)}")
    print(f"- Всего слов: {len(words)}")
    print(f"- Уникальных слов: {len(unique_words)}")
    print(f"- Стоп-слов: {len(stop_words)}")
    print(f"- Средняя длина слова: {avg_word_length:.2f} символов")
    
    print(f"\n=== МОРФОЛОГИЧЕСКИЙ АНАЛИЗ ===")
    print(f"- Распределение частей речи:")
    for pos, count in pos_counts.most_common():
        print(f"  • {pos}: {count} ({count/len(words)*100:.1f}%)")
    print(f"- Уникальных лемм: {len(unique_lemmas)}")
    print(f"- Лемматизированное соотношение: {len(unique_lemmas)/len(unique_words):.2f}")
    
    print(f"\n=== СИНТАКСИЧЕСКИЙ АНАЛИЗ ===")
    print(f"- Синтаксические зависимости (топ-5):")
    for dep, count in dependencies.most_common(5):
        print(f"  • {dep}: {count}")
    
    if has_noun_chunks:
        noun_chunks = list(doc.noun_chunks)
        print(f"- Именных групп (noun chunks): {len(noun_chunks)}")
    else:
        print(f"- Именные группы (noun chunks): не поддерживаются для русского языка")
    
    print(f"\n=== ИМЕНОВАННЫЕ СУЩНОСТИ ===")
    print(f"- Всего сущностей: {len(entities)}")
    if entity_counts:
        print(f"- Типы сущностей:")
        for label, count in entity_counts.most_common():
            print(f"  • {label}: {count}")
    
    print(f"\n=== ЛЕКСИЧЕСКОЕ РАЗНООБРАЗИЕ ===")
    print(f"- TTR (соотношение тип-токен): {ttr:.3f}")
    print(f"- MTLD (упрощенный): {mtld:.2f}")
    
    print(f"\n=== СТРУКТУРА ТЕКСТА ===")
    print(f"- Количество предложений: {sentence_count}")
    print(f"- Средняя длина предложения: {avg_sentence_length:.2f} токенов")
    print(f"- Вопросительных предложений: {question_count} ({question_count/sentence_count*100:.1f}% от общего)" if sentence_count else "- Вопросительных предложений: 0")
    print(f"- Восклицательных предложений: {exclamation_count} ({exclamation_count/sentence_count*100:.1f}% от общего)" if sentence_count else "- Восклицательных предложений: 0")
    
    print(f"\n=== ЧИТАБЕЛЬНОСТЬ ===")
    print(f"- Индекс Флеша-Кинкейда: {flesh_kincaid:.2f}")
    print(f"- Процент длинных слов: {long_words_percent:.2f}%")
    
    print(f"\n=== ЧАСТОТНЫЙ АНАЛИЗ ===")
    print(f"- Наиболее частотные слова:")
    for word, count in most_common_words:
        print(f"  • {word}: {count}")
    
    if len(sentences) > 1:
        print(f"\n=== СЕМАНТИЧЕСКАЯ СВЯЗНОСТЬ ===")
        coherence_scores = []
        for i in range(len(sentences)-1):
            if len(sentences[i]) > 0 and len(sentences[i+1]) > 0:
                try:
                    if sentences[i].vector_norm > 0 and sentences[i+1].vector_norm > 0:
                        sim = sentences[i].similarity(sentences[i+1])
                        coherence_scores.append(sim)
                except:
                    pass
        
        if coherence_scores:
            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            print(f"- Средняя семантическая связность между предложениями: {avg_coherence:.3f}")

text = """
В современной лингвистике наблюдается увеличение интереса к вопросам языкового моделирования текстового мира художественного произведения на основе дейксиса (Н.А. Серебрянская [1], С.А. Пушмина [2], О.Г. Мельник [3]). Дейксис рассматривается как одно из основных понятий прагматики. Взаимосвязь прагматического значения высказывания с контекстом также имеет большое значение для понимания ситуации общения. Определение дейксиса, предложенное Дж. Лайонзом, подчеркивает эту взаимосвязь: «Под дейксисом подразумевается локализация и идентификация лиц, объектов, событий, процессов и действий, о которых говорится или на которые ссылаются относительно пространственно-временного контекста, создаваемого и поддерживаемого актом высказывания и участием в нем, как правило, одного говорящего и, по крайней мере, одного адресата [4. С. 539]. Целью данной статьи является анализ роли дейксиса в интерпретации художественного текста на примере романа Н. Геймана «Никогде». Н.А. Серебрянская отмечает, что «дейксис проецируется на художественный текст через универсальные смыслы человек, пространство, время; через точку зрения наблюдателя, имеющего определенную позицию в пространстве и во времени; через оппозицию близко – далеко по отношению к тексту со стороны наблюдателя; через сфокусированное значение антропоцентризма в тексте, через участие в создании хронотопа и так далее» [1. С. 24]. Традиционно выделяют три типа дейксиса: лица, места и времени. В дополнение к этим типам Ч. Филмор [5. С. 103–107] выделяет также дейксис дискурса и социальный дейксис. Р. Лакофф [8. С. 346] обращает внимание на эмоциональный дейксис. Б. Крик [7. С. 23] называет эти три дополнительных типа «маргинальными категориями». Ч. Филмор также выделяет «жестовое», «символическое» и «анафорическое» использование дейктических элементов, причем под «символическим» подразумевается указание на ненаблюдаемые элементы речевой ситуации [5]. В этой связи рассмотрим примеры высказываний, где Н. Гейман акцентирует внимание на физическом аспекте коммуникационной ситуации, сопровождающей речевой акт и усиливающем его прагматическое значение: 1. «The Angelus is through there», announced Door, interrupting his reverie, pointing to the direction from which the music was coming [8. С. 70]. 2. The girl called Door passed the paper to Richard. «Here», she said. «Read this» [8. C. 16]. 3. «It's me little flag,» he said, pointing to the handkerchief [8. С. 128]. Каждый из этих примеров включает оба элемента чистого дейксиса: саму дейктическую единицу и сопроводительный жест. Наречие there в примере (1), произнесенное Дверью, подкрепляется жестовым указанием в сопровождении существительного direction. В примере (2) высказывание сопровождается физическим движением. Такая комбинация дейктика с описанием движения привлекает внимание читателя сильнее, чем одни лишь дейктики. Сопроводительные жесты подчеркиваются дополнительным лингвистическим описанием, таким как указание на предмет, как в примере (3). Эти примеры чистого дейксиса представляют собой своего рода письменное свидетельство, позволяющее читателю проследить направление указания. В отличие от жестового дейксиса, для определения символического дейксиса не требуются экстралингвистические указатели. Для устранения неоднозначности необходимы лишь общие знания о временных и местных параметрах конкретной речевой ситуации и о участниках общения. Так, знаний об адресате вполне достаточно, чтобы интерпретировать дейктическую референцию местоимений you и your: 4. You really could get lost in your own backyard, Richard [8. C. 5]. Из контекста читатель понимает, что Джессика обращается к Ричарду, следовательно, для определения референции не требуется непосредственное указание. В примере (5) также не требуется визуальное объяснение для интерпретации дейктической референции here: 5. You put that girl down and come back here this minute. Or this engagement is at an end as of now [8. C. 10]. Поскольку интерпретация here соотносится с местоположением Джессики, достаточно понимать из контекста, где она находится в момент высказывания. Читатель знает, что действие происходит на улице у входа в ресторан, таким образом, местоименное наречие кореферентно с этим местом. Существуют дейктические выражения, такие как here and there, this and that, now and then, которые могут использоваться недейктически, так как полностью перешли в разряд идиом. Их значение никак не зависит от личности говорящего, времени или места общения: 6. Lights flickered dimly, here and there in the walls, beside the paths, and, far, far below them, tiny fires were burning [8. C. 108]. В данном случае идиома обозначает отсутствие упорядоченности, «в разных местах». Личный дейксис характеризуется свойством постоянного изменения во времени. С одной стороны, референт дейктика «я» изменяется вместе со сменой ролей «говорящий – адресат» в процессе общения: 7. «You look like a drowned rat,» said someone. – «You've never seen a drowned rat,» said Richard [8. С. 2]. С другой стороны, изменчивость можно рассматривать с философской точки зрения в том смысле, что «я, здесь и сейчас» отличается от «я, там и тогда». Ричард попадает в Нижний Лондон, пройдя невидимую границу между двумя мирами. Его личность расщепляется, он ощущает, как будто бы существует другой Ричард в каждом мире: 8. The old Richard, the one who had lived in what was now the Buchanans' home, would have crumbled at this point, apologized for being a nuisance, and gone away. Instead, Richard said, «Really? Nothing you can do about it? … Now, I happen to think, and I'm sure my lawyer will also think, that there is a great deal you can do about it» [8. С. 133]. В этой связи уместно упомянуть теорию возможных миров и жесткие/нежесткие десигнаторы С. Крипке. Возможными мирами С. Крипке называются миры предполагаемые, а не открытые (stipulated, not discovered) [9. С. 267], т.е. некий наблюдатель решает, где они начинаются и определяет границу между возможными мирами и реальным миром. Таким образом, мы можем предположить, что Нижний Лондон – это возможный мир, где все правила резко отличаются от знакомых норм реального мира: 9. He wondered how normal London-his London-would look to an alien, and that made him bold [8. С. 42]. В рамках теории о возможных мирах С. Крипке [9. С. 269–271] рассматривается проблема идентичности в возможных мирах (identity across possible worlds) или «трансмировой идентичности» (transworld identification). Перемещение Ричарда в Нижний Лондон символически можно интерпретировать как переход в возможный мир, который отличается от Верхнего Лондона. В соответствии с мнением С. Крипке можно предположить, что имя Ричард обозначает один и тот же объект в любом из возможных миров и является жестким десигнатором, определяющим жесткую референцию [9. С. 277], т.е. он относится к одной и той же индивидуальности в любом из возможных миров. По замечанию Д. Льюиса [10. C. 52], реальный мир – это тоже возможный мир, в котором мы находимся, следовательно, два существующих Ричарда в двух мирах представляют один и тот же референт, хотя Ричард и не всегда в этом уверен: 10. «It's like I've become some kind of non-person» [8. С. 42]. Приведенный пример демонстрирует, что личный дейксис, относящийся к одному и тому же человеку, не обязательно отражает чувство собственной неизменности. Таким образом, дейктик «я» действительно является транзитной переменной, прагматическое значение которой зависит от контекста. Важной особенностью личного дейксиса является его контекстуальная зависимость от центра координат (Origo). Референция дейктического выражения может быть установлена только если определен центр ориентации. Рассмотрим примеры: 11. «I'll meet you at your place», said Jessica [8. С. 6]. 12. «I don't know what you think you're doing», said Richard. «But if you two don't get out of my apartment this minute, I'm phoning the police» [8. С. 13]. В отрыве от контекста референция I в примере (11) могла бы варьироваться в зависимости от того, кто произносит фразу. Однако в этом случае контекст определяет говорящего. Поскольку говорящий – это Джессика, центр ориентации находится там, где она располагается, и референт I относится к ней. Анализируя what you think you're doing в широком контексте (12), понимаем, что центр дейктического поля – это Ричард. Произнесенное им местоимение you обозначает связь с эгоцентричной точкой ориентации. Контекст уточняет, что you относится к двум бандитам. Анализируя личный дейксис, читатель часто сталкивается с неоднозначностью референции, которую можно прояснить за счет контекста. Это часто объясняется неочевидным распределением ролей, как в следующем примере: 13. The marquis… stood in front of Varney, who looked obscenely pleased with himself... «Well», said the marquis de Carabas. «We're all very impressed with your skill». «I had heard», said a female voice, «that you had put out a call for bodyguards. Not for enthusiastic amateurs». «Varney», said Varney, affronted, «is the best guard and bravo in the Underside. Everyone knows that». The woman looked at the marquis. «You've finished the trials?» she asked. «Yes», said Varney. «Not necessarily», said the marquis. «Then», she told him. «I would like to audition» [8. С. 46]. Чтобы верно интерпретировать дейктик, важно учитывать, как роли участников беседы грамматикализуются в данной ситуации. Из примера видно, что говорящий – это Охотница, но не совсем понятно, кто получатель (мишень) высказывания, а кто просто наблюдатель (слушатель). Если мы рассмотрим только непосредственный контекст, может показаться, что you является референцией к получателю Варни, поскольку он первым отвечает на заданный вопрос. Однако если мы рассмотрим более широкий контекст, станет ясно, что различие между получателем сообщения и наблюдателем грамматикализовано при помощи дейктиков не самым очевидным способом. Дальнейшее чтение подскажет правильную референцию, вопрос Охотницы адресован маркизу де Карабасу. Другой важной особенностью личных дейктиков, рассматриваемой в рамках проблемы неоднозначности, является различие между инклюзивным/эксклюзивным употреблением местоимения we (we-inclusive-of addressee and we-exclusive-of addressee [11. C. 69]. Из названия понятно, что первое из них включает адресата в референцию, тогда как второе не включает. Английская грамматика не позволяет прямо указывать на включение или исключение адресата из референции, однако его можно легко вывести из контекста: 14. She threw her arms around his chest and hugged him, tightly. «And we will try to get you back home again», she said. «Promise. Once we've found what I'm looking for» [8. С. 50]. В данном примере, если не анализировать контекст, использование местоимения we можно рассматривать и как включающее, и как исключающее адресата. Однако, учитывая контекст, мы делаем вывод, что это инклюзивное использование. Дверь окончательно принимает решение взять Ричарда с собой и пытается ободрить его, вовлекая в совместные действия. Дейксис места указывает на местоположение относительно центра высказывания, т.е. лингвистическими средствами указывает на положение говорящего в трехмерном пространстве, а также на его/ее отношение к местоположению других участников беседы: 15. Anaesthesia hesitated and then turned left [8. С. 35]. 16. «You're not wanted here, de Carabas. Get away. Clear off» [8. С. 19]. Эти примеры показывают, как дейктический центр привязан к говорящему, делая его центром коммуникации. В примере (15) left указывает на направление поворота относительно Анестезии; в примере (16) here относится к жилищу бродяги, на чью территорию пожаловали незваные гости. Знание о местоположении говорящего очень важно для читателя, так как оно помогает ориентироваться при помощи указательных местоимений this/these and that/those, указывающих на проксимальное и дистальное положение: 17. «Allow me to make introductions. I am Mister Croup, and this gentleman is my brother, Mister Vandemar» [8. С. 12]. 18. «One portion of vegetable curry, please», said Richard, to the woman at the curry stall. «And, um, I was wondering. The meat curry. What kind of meat is it, then?» The woman told him. «Oh», said Richard. «Right. Um. Better just make that vegetable curries all round» [8. С. 103]. В примере (17) Мистер Круп указывает на дейктическую близость, тогда как в примере (18) Ричард отдает предпочтение местоимению that, так как он находится по другую сторону от стойки, за которой стоит его собеседница. Аналогичным образом интерпретация дейктиков here и there также зависит от положения говорящего: 19. The earl beckoned to Door. «Come here», he said. «Come-come-come. Let me look at you» [8. С. 57]. 20. «There's someone else out there. Mister Croup?» There was a dark shimmer where Mr. Croup had been, and he was there no longer [8. С. 118]. В примере (19) here означает пространство вокруг говорящего в момент произнесения высказывания. В примере (20) there, наоборот, указывает на удаленность от говорящего, местоположение адресата. Вышеприведенные примеры демонстрируют различные способы грамматикализации дейксиса места, основной принцип которого основывается на оппозиции близости/дальности дейктических выражений. Временной дейксис выражается при помощи дейктиков, чья референция может быть установлена только по отношению ко времени высказывания. Дейктическая функция у наречия now является ведущей. Оно обозначает время, непосредственно предшествующее или непосредственно следующее за точкой отсчета – моментом речевого акта – или абсолютным временем протекания действий, событий, явлений, процессов объективной действительности: 21. «Right now we're looking for an angel named Islington» [8. С. 50]. В определенном контексте значение now может быть приближено к then, что подчеркивается употреблением прошедшего времени: 22. «So where were you?» he asked. «Just now?» – «I was here», she said [8. С. 15]. Данный пример демонстрирует близкую связь между языком, непосредственным контекстом и контекстом дискурса в целом, представленным в феномене временного дейксиса. Другим интересным примером зависимости временного дейксиса от контекста и точки отсчета является функционирование наречий today и tomorrow: 23. «I saved his life three times today, crossing the bridge, coming to the market» [8. С. 47]. Будучи индексальными символами, дейктики имеют основной компонент значения, называемый семантическим значением, и переменный компонент, называемый прагматическим значением, которое определяет референт дейктика в конкретном контексте. Дейктическое слово today указывает на темпоральную референцию, близкую к моменту высказывания. Согласно словарю [12] основной семантический компонент значения наречия today определяется как «the day on which you are speaking or writing». Определение прагматического компонента значения этого дейктика представляется проблематичным. Поскольку читателю неизвестно время кодирования, он/она не могут определить точку реального времени, которая будет означать точную темпоральную локацию «сегодня». Читатель может только предположить, что наречие today в данном контексте относится к некоему неопределенному моменту в той части дня, которая еще не истекла. В следующем примере дейктик today имеет более широкое значение, которое дается в словаре под цифрой 2, – «the present period of history»: 24. «Me mam told me not to go marrying outside, but I was young and beautiful, although you'd never credit it today, and I followed my heart» [8. С. 1]. Сходные проблемы возникают с интерпретацией темпорального дейктического выражения tomorrow, как в следующем примере: 25. «Of course. We'll have all of this rubbish cleaned out of here tomorrow, no problem» [8. С. 24]. Дейктик tomorrow означает удаленность от временной референции говорящего. Семантически он указывает на день в будущем, который означает суточный промежуток, следующий за промежутком, означающим время кодирования. Это значение остается постоянным в любых случаях использования этого наречия. Однако точное время кодирования (25) неизвестно читателю. Это может происходить и в течение двадцати четырех часов, и на протяжении одной минуты перед указанным моментом. Следовательно, определить прагматическое значение дейктика tomorrow в контексте примера (25) довольно трудно. Дейксис дискурса – это языковая единица, которая имеет референт в дискурсе. Он часто напоминает анафору, чей референт совпадает с ранее упомянутым объектом. К. Элих отмечает, что анафора является непрямой референцией [13. С. 316], с лингвистической точки зрения это перекрестная референция, поскольку она относится к ранее упомянутым словам, а те, в свою очередь, относятся к объектам или индивидам в реальном мире. Причина частого смешения анафорического и дейктического употребления местоимений объясняется тем фактом, что в соответствии с традиционным определением местоимение всегда относится к своему антецеденту [6. P. 668], что понимается как выражение, предшествующее употреблению местоимения в дискурсе: 26. He spotted the towel on the chair in the hall, and he leaned out and grabbed it [8. С. 24]. В примере (26) местоимение it используется анафорически, поскольку оно кореферентно своему антецеденту towel. В следующем примере антецедент не так очевиден: 27. «She's not here anymore. And I don't know where she is», – «We know that, Mister Mayhew» [8. С. 25]. Это пример непрямой анафоры, потому что антецедент не поддается прямому определению, а подразумевается. Теория непрямой анафоры является дискуссионным вопросом, потому что принятое в лингвистике определение анафоры охватывает случаи как прямой, так и непрямой референции. Для распознавания анафоры важно анализировать ее употребление в контексте. Без опоры на контекст большинство дейктических выражений двусмысленны, как в следующем примере: 28. Then it turned to Richard. «And you? What do you want, Richard Mayhew?» Richard shrugged. «I want my life back. And my apartment. And my job», «That can happen», said the angel [8. С. 76]. В примере (28) that может анафорически относиться либо к некоторым перечисленным объектам, либо ко всем сразу. Однако более широкий контекст снимает эту неоднозначность: ангел заявляет о своем всемогуществе, следовательно, речь идет обо всех перечисленных объектах. В отличие от анафоры дискурсивные дейктические выражения определяют их референцию не за счет выделения антецедента, а при помощи указания на часть дискурса. Дейксис дискурса отсылает к экстралингвистическим объектам, что означает, по образному выражению Б. Крик, он «пересекает границы предложения» [7. С. 68]. Для того чтобы сопоставить дейксис дискурса и анафору, рассмотрим следующий пример: 29. The big man simply pushed past him and walked into the apartment, a wolf on the prowl. Richard ran after him. «What do you think you are doing? Will you stop that? Get out» [8. С. 13]. Указательное местоимение that в примере (29) привязано к ситуации общения. В отличие от примера (20) функция этого местоимения не анафорическая, поскольку она не кореферентна с каким-либо антецедентом, местоим
"""

analyze_text(text)
