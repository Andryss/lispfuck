## Отчет

* Кривошеев Андрей Александрович, P33111
* `lisp | acc | harv | hw | instr | binary | stream | mem | cstr | prob5 | 8bit`
* Без усложнения


#### Пояснение варианта

---

* `lisp` -- Язык программирования. Синтаксис -- синтаксис языка Lisp. S-exp.
  * требуется поддержка функций и/или процедур;
  * необходимо объяснить, как происходит отображение сложных выражений на регистры и память;
  * необходимо продемонстрировать работу транслятора в случае, если количества регистров недостаточно для реализации алгоритма.
  * Любое выражение (statement) -- expression. Примеры корректного кода (с точностью до ключевых слов):
    * (print (if (= 1 x) "T" "F"))
    * (setq x (if (= 1 x) 2 3))
    * (setq x (loop ...))
    * (print (seq x 13))
  * Необходимо объяснить и продемонстрировать, что любое выражение (statement) -- expression.

* `acc` -- Архитектура -- система команд должна быть выстроена вокруг аккумулятора.
  * Инструкции -- изменяют значение, хранимое в аккумуляторе.
  * Ввод-вывод осуществляется через аккумулятор.

* `harv` -- Архитектура организации памяти -- Гарвардская архитектура.

* `hw` -- Control Unit -- hardwired. Реализуется как часть модели.

* `instr` -- Точность модели -- процессор необходимо моделировать с точностью до каждой инструкции (наблюдается состояние после каждой инструкции).

* `binary` -- Представление машинного кода -- бинарное представление.
  * Требуются настоящие бинарные файлы, а не текстовые файлы с 0 и 1.
  * Требуется отладочный вывод в текстовый файл вида:
    ```text
    <address> - <HEXCODE> - <mnemonica> 
    20 - 03340301 - add #01 <- 34 + #03
    ```
    
* `stream` -- Ввод-вывод
  * Ввод-вывод осуществляется как поток токенов. Есть в примере. Логика работы:
    * при старте модели у вас есть буфер, в котором представлены все данные ввода (['h', 'e', 'l', 'l', 'o']);
    * при обращении к вводу (выполнение инструкции) модель процессора получает "токен" (символ) информации;
    * если данные в буфере кончились -- останавливайте моделирование;
    * вывод данных реализуется аналогично, по выполнении команд в буфер вывода добавляется ещё один символ;
    * по окончании моделирования показать все выведенные данные;
    * логика работы с буфером реализуется в рамках модели на Python.

* `mem` -- Ввод-вывод ISA -- memory-mapped (порты ввода-вывода отображаются в память и доступ к ним осуществляется штатными командами),
  * отображение портов ввода-вывода в память должно конфигурироваться (можно hardcode-ом).

* `cstr` -- Поддержка строк -- Null-terminated (C string)

* `prob5` -- Алгоритм -- Smallest multiple. [Project Euler. Problem 5](https://projecteuler.net/problem=5)

* `8bit` -- машинное слово -- 8 бит (как для памяти команд, так и для памяти данных, если они разделены).

---


### Язык программирования

Форма Бэкуса-Наура:

```text

<program> ::= <statement> | <statement> <program>
<statement> ::= <function_definition> | <common_statement>

<function_definition> ::= "(" <function_name> "(" <function_arguments> ")" <function_body> ")"
<function_name> ::= <identifier>
<function_arguments> ::= "" | <function_argument> | <function_argument> <function_arguments>
<function_argument> ::= <identifier>
<function_body> ::= <argument>

<common_statement> ::= "(" <statement_name> <arguments> ")"
<statement_name> ::= <boolean_operation> | <math_operation> | <identifier>
<boolean_operation> ::= "=" | ">="
<math_operation> ::= "+" | "-" | "*" | "/"
<arguments> ::= "" | <argument> | <argument> <arguments>
<argument> ::= <literal> | <variable> | <common_statement> 

<literal> ::= <number> | <string>
<number> ::= [ "-" ] <unsigned_number>
<unsigned_number> ::= <digit> | <digit> <unsigned_number>
<digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
<string> ::= '"' <string_letters> '"'
<string_letters> ::= "" | <string_letter> | <string_letter> <string_letters>
<string_letter> ::= <any symbol except: '"'>

<variable> ::= <identifier>
<identifier> ::= <identifier_letters>
<identifier_letters> ::= <identifier_letter> | <identifier_letter> <identifier_letters>
<identifier_letters> ::= "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" 
  | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" 
  | "V" | "W" | "X" | "Y" | "Z" | "a" | "b" | "c" | "d" | "e" | "f" | "g" 
  | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" 
  | "t" | "u" | "v" | "w" | "x" | "y" | "z" | "_"

```

Код состоит из последовательности выражений, заключенных в скобки. Выражение состоит из названия и агрументов, следующих 
друг за другом через пробел. В качестве аргументов могут выступать также выражения, переменные или числовые (`123`) или 
строковые литералы (`"str"`). Пробельные символы и переносы строк игнорируются.

Любое выражение/переменная/литерал имеют возвращаемое значение (кроме объявления функции).

Объявление функции выглядит, как обычное выражение, включающее название-аргументы-телою Тело функция - ровно одно 
выражение.

Типов данных два: 64-битное знаковое число, строка. Соответственно, каждое выражение возвращает значение одного из двух 
типов.

На уровне языка поддерживаются математические выражения `+` - сложение, `-` - вычитание, `*` - умножение, 
`/` - целочисленное деление, `mod` - остаток от деления.

Поддерживаются операторы сравнения `=` - равно, `>=` - больше или равно. Возвращаемое значение 1 - истина, 0 - ложь.

Переменные можно определить при помощи функции `set` (`(set <label> <val>)`). Возвращает установленное значение. 
Область видимости данных - глобальная.

Также поддерживается условная конструкция `if` (`(if <cond> <opt1> <opt2>)`). Истина определяется по значению `cond`, 
равно 0 - будет выполнен код `opt2`, иначе `opt1`.

Определены функции для работы с вводом выводом: `printi`, `prints`, `read`. Функции вывода возвращают количество 
выведенных символов, а функция чтения - прочитанную строку.

Код выполняется последовательно сверху вниз. Внутри выражений код выполняется *справа налево*.


### Организация памяти

Память разделена на память команд (инструкций) и память данных.

Память инструкций имеет примерно следующий вид:

```text

    +------------------------+
    | 0: br n                |
    | 1: <func1 code>        |
    | ...                    |
    | k: <func2 code>        |
    | ...                    |
    | n: <main code>         |
    | ...                    |
    | m: halt                |
    +------------------------+
    
```

Машинное слово - 32 бита (8 бит - код операции, 4 бит - тип адресации, 20 бит - значение для адресации).

Первая инструкция - переход к основной программе (которая была странслирована из исходного кода, написанного программистом).
Далее хранятся друг за другом используемые программой функции. После всех функции расположены инструкции основной программы.
Последняя инструкция - `halt`.

Память данных имеет примерно следующий вид:

```text

    +------------------------+
    | 0: const 1             |
    | 1: const 2             |
    | ...                    |
    | v+0: variable 1        |
    | v+1: variable 2        |
    | ...                    |
    | b+0: anon buffer 1     |
    | b+1: anon buffer 2     |
    | ...                    |
    | ...                    |
    | n: stack start         |
    +------------------------+   
    
```

Машинное слово - 64 бита.

Сначала располагаются константы (числовые - 1 слово, строковые - количество символов + 1 слов). Далее находятся
переменные (по 1 машинному слову). Далее выделены анонимные буферы для некоторых команд, которые представляют
несколько размеченных друг за другом слов (например, `read` выделяет буфер для чтения данных из ввода). Буферы всегда 
инициализируются 0. В конце памяти данных находится место для стека, он "растет" от конца памяти к началу.

Программист с регистрами и памятью напрямую не работает.

Строковые литералы всегда считаются константами и записываются в область констант. Числовые литералы считаются
константой, если не влазят в 20 бит (то есть больше 2^19-1 или меньше -2^19). Остальные числа будут загружены в 
регистры при помощи непосредственной адресации.

Порядок расположения констант, переменных, буферов определяется порядком встречи в коде.

Во время трансляции исходной программы если заявляется необходимость выделения константы/переменной/буфера, то в
машинном коде временно записывается символ (идентификатор) выделенной памяти. На конечном этапе трансляции происходит 
маппинг всех запрошенных констант/переменных/буферов на память, а все идентификаторы в машинном коде заменяются на
непосредственный адрес.

В стек ничего не отображается.

Поскольку архитектура процессора - аккумуляторная, то все переменные отображаются в память, а при необходимости
обращения в аккумулятор загружается их значение. При вызове функций первый аргумент (если есть) передается через 
аккумулятор, а последующие (если есть) передаются через стек.


### Система команд

Типов данных два: 64-битное знаковое число, строка. Внутри процессора число представляет собой свое значение, а строка -
указатель по абсолютному адресу на первый символ.

Регистры:

| Название   | Разрядность | Описание                                                                    |
|------------|-------------|-----------------------------------------------------------------------------|
| `acr`      | 64          | аккумулятор                                                                 |
| `ipr`      | 64          | указатель на адрес исполняемой инструкции                                   |
| `inr`      | 32          | содержит текущую исполняемую инструкцию                                     |
| `bur`      | 64          | буферный регистр, используется для проведения операций между arc, ipr, inr  |
| `adr`      | 64          | регистр адреса, используется для чтения/записи                              |
| `dar`      | 64          | регистр данных, используется для чтения/записи                              |
| `spr`      | 64          | указатель на вершину стека                                                  |
| `flr`      | 4           | регистр флагов, содержит флаги по результатам операции (NZVC)               |

Доступ к памяти данных осуществляется при помощи регистров `adr`, `dar`. Для чтения необходимо задать адрес ячейки для 
чтения. Для записи необходимо задать адрес и значение для записи.

Ввод/вывод отображается на память. Для вывода необходимо выполнить запись в ячейку памяти по определенному адресу. 
Для ввода необходимо выполнить чтение определенной (другой) ячейки памяти.

Имеется 5 типов адресации:
  * непосредственная
  * абсолютная
  * относительная (`ipr`)
  * относительная (`spr`)
  * относительная косвенная (`spr`)

Инструкции делятся на 3 типа: без аргумента, адресные, по значению.
Инструкции без аргумента не требуют адресации. Адресные используют аргумент адреса с которым будет выполнено действие.
Инструкции по значению используют адресацию, для указания на адрес значения, с которым будет выполнено действие.

Набор инструкций:

| Инструкция    | Описание                                                          |
|---------------|-------------------------------------------------------------------|
| `noop`        | ничего не выполняется                                             |
| `halt`        | останов                                                           |
| `ld <val>`    | загрузка значения из памяти в аккумулятор                         |
| `st <addr>`   | запись значения аккумулятора в память                             |
| `call <addr>` | вызвать функцию по адресу (адрес возврата положить на стек)       |
| `ret`         | возвращение из функции (адрес возврата снять со стека)            |
| `push`        | положить значение аккумулятора на стек                            |
| `pop`         | снять значение со стека и записать в аккумулятор                  |
| `popn`        | снять значение со стека без записи куда-либо (spr--)              |
| `cmp <val>`   | установить флаги по результату операции вычитания из аккумулятора |
| `bre <addr>`  | переход по адресу если равно (Z == 1)                             |
| `brge <addr>` | переход по адресу если значение больше или равно (N == V)         |
| `br <addr>`   | безусловный переход по адресу                                     |
| `inc <addr>`  | инкремент значения по адресу (без изменения аккумулятора)         |
| `dec <addr>`  | декремент значения по адресу (без изменения аккумулятора)         |
| `mod <val>`   | деление по модулю аккумулятора и значения                         |
| `add <val>`   | сложить значение с аккумулятором                                  |
| `sub <val>`   | вычесть из аккумулятора значение                                  |
| `mul <val>`   | умножить аккумулятор на значение                                  |
| `div <val>`   | поделить аккумулятор на значение (целочисленное деление)          |
| `inv`         | инвертировать значение аккумулятора                               |

где:
  * `<val>` - аргумент будет интерпретирован как значение (значение по адресу), с которым необходимо выполнить операцию
  * `<addr>` - аргумент будет интерпретирован как адрес, с которым необходимо выполнить операцию

Поток управления:
  * вызов `call` или возврат `ret` из функции
  * условные `bre`, `brge` и безусловные `br` переходы
  * инкремент `ipr` после любой другой инструкции

Машинный код сериализуется в список 32 битных слов. Одно слово - одна инструкция. Адресация инструкций с 0 по порядку.

Кодирование инструкций:
  * 8 бит - код операции
  * 4 бит - тип адресации (для команд с аргументом)
  * 20 бит - значение для адресации (для команд с аргументом)
  * (для команд без аргумента 24 младших бита равны 0)

Типы данных в модуле [isa](isa.py):
  * Opcode - перечисление кодов операций
  * AddressType - перечисление типов адресации
  * Address - структура объединение типа адресации и значения
  * Term - структура для описания одной инструкции

Стандартная библиотека реализована в модуле [stdlib](stdlib.py).


### Транслятор

Интерфейс командной строки: 

```text
usage: translator.py [-h] [--output output_file] [--debug debug_file] [--verbose] [--optimize_math] [--optimize_pop_push] source_file

Translates lispfuck code into en executable file.

positional arguments:
  source_file           file with lispfuck code

options:
  -h, --help            show this help message and exit
  --output output_file, -o output_file
                        file for storing a binary executable (default: output)
  --debug debug_file, -d debug_file
                        file for storing a binary executable explanation info (default: debug.txt)
  --verbose, -v         print verbose information during conversion
  --optimize_math       optimize math operations (no guarantee of evaluation strategy)
  --optimize_pop_push   remove pop and push instructions placed next to each over
```

Реализовано в модуле: [translator](translator.py)

Этапы трансляции:
  * чтение исходного кода
  * трансформация текста в последовательность токенов с тегами
  * преобразование токенов в абстрактное дерево (выражение - вершина, литерал/идентификатор - лист)
  * вычленение Statement на основе дерева и валидация и проверка корректности использования функций
  * генерация машинного кода
  * запись полученного кода в бинарный файл 
  * запись полученного кода в текстовый файл

Правила генерации машинного кода:
  * каждый Statement должен к концу исполнения должен установить в аккумулятор возвращаемое значение, а также
    вернуть spr в то состояние, в котором он был в начале исполнения
  * при вызове функции вызывающая сторона обязана установить все требуемые аргументы, а также после возврата
    снять со стека все положенные аргументы
  * Statement содержащий переменную или литерал должен загрузить в аккумулятор нужное значение
  * Statement с несколькими аргументами для вычисления производят генерацию кода вычисления с последнего к первому
    (справа налево)
  * при необходимости выделения места в памяти под константу/литерал/буфер, происходит обращение в специальный
    класс, который запоминает все обращения и выдает идентификаторы для последующей генерации памяти данных
    и замены символов на непосредственные адреса
  * при трансляции кода для функции каждый Statement должен учитывать то, где сейчас находятся (в аккумуляторе, в стеке)
    аргументы и помещать их на стек в случае необходимости, а также отслеживать их актуальное положение 
    (смещение относительно spr)
  * после завершения генерации кода, все символы в инструкциях заменяются на адреса в зависимости от их отображения
    в память


### Модель процессора

Интерфейс командной строки:

```text
usage: machine.py [-h] [--input input_file] [--verbose] binary_file

Execute lispfuck binary executable file.

positional arguments:
  binary_file           binary executable file

options:
  -h, --help            show this help message and exit
  --input input_file, -i input_file
                        file with input data for executable (default: empty file)
  --verbose, -v         print verbose information during execution
```

Реализовано в модуле: [machine](machine.py)

#### Data path

```text

                      latch --->+-------+                               +-------+<--- latch   
                                |  acr  |---------+           +---------|  adr  |             
                   +----------->+-------+         |           |         +-------+<-----------+
                   |                              |           |                              |
                   |  latch --->+-------+         |           |         +-------+<--- latch  |
                   |            |  ipr  |-------+ |           | +-------|  dar  |   +-----+  |
                   +----------->+-------+       | |           | |       +-------+<--| MUX |--+
                   |                            | |           | |           sel --->+-----+  |
                   |            +-------+       | | 0       0 | |       +-------+<--- latch  |
                   |            |  inr  |-----+ | | |       | | | +-----|  spr  |            |
                   |            +-------+     | | | |       | | | |     +-------+<-----------+
                   |                          v v v v       v v v v                          |
                   |                        +---------+   +---------+                        |
                   |                sel --->|   MUX   |   |   MUX   |<--- sel                |
                   |                        +---------+   +---------+                        |
                   |                             |             |                             |
                   |                             v             v                             |
         latch     |           inv_left --->+---------+   +---------+<--- inv_right          |
           |       |          extend_20 --->|          \_/          |<--- (+1)               |
           v       |                        |                       |                        |  
       +-------+   |             opcode --->|          ALU          |                        |
       |  flr  |<----------------------------\                     /                         |
       +-------+   |                          +-------------------+                          |
                   |                                    |                                    |
                   +------------------------------------+------------------------------------+

```

#### Взаимодействие с памятью данных

```text
            
                                 read     write
                                   |        |
                                   v        v
              +-------+          +------------+
              |  adr  |--------->|            |<------- input
              +-------+          |            |
                                 |    data    |-------> output
                                 |   memory   |
              +-------+          |            |
   latch ---->|  dar  |--------->|            |-------+
              +-------+          +------------+       |
                   ^                                  |
                   |   +-----+                        |
                   +---| MUX |------------------------+
                       +-----+
                          ^
                          |
                         sel
                         
```

Реализован в классе `DataPath`.

Сигналы:
  * `read_data_memory` - чтение данных по адресу `adr` в `dar`:
    * из памяти данных (`dar <- mem[adr]`)
    * из порта ввода `input`:
      * извлечь из входного буфера токен и подать его на выход
      * если буфер пуст - исключение
  * `write_data_memory` - запись данных `dar` по адресу `adr`:
    * в память данных (`mem[adr] <- dar`)
    * в порт вывода `output`:
      * записать значение `dar` в буфер вывода
  * `latch_acr`, `latch_ipr`, `latch_bur`, `latch_adr`, `latch_spr` - записать значение с выхода ALU в регистр
  * `sel_dar` - выбрать значение с выхода ALU или из памяти данных для записи в `dar`
  * `latch_dar` - записать выставленное значение в `dar`
  * `latch_flr` - записать значения флагов операции суммы из ALU в `flr`
  * `sel_left`, `sel_right` - выбрать значения для левого и правого входов ALU
  * `alu_opcode` - выбор операции для осуществления на ALU (sum, mul, div, mod)
  * `alu_inv_left`, `alu_inv_right` - инвертировать левый вход, правый вход ALU соответственно
  * `alu_extend_20` - расширить знак 20 бит левого входа ALU
  * `alu_plus_1` - прибавить 1 к операции суммы ALU


#### ControlUnit (взаимодействие с памятью инструкций)

```text

                                                                   +---------+
                                                               +-->|  step   |--+                 input   output
                                                   latch       |   | counter |  |                   |       ^
         latch             +--------------+          |         |   +---------+  v                   v       |
           |      read --->|              |          v         |     +---------------+  signals   +-----------+
           v               | instructuion |       +-------+    +-----|  instruction  |----------->| DataPath* |
        +-------+          |    memory    |------>|  inr  |--------->|    decoder    |            +-----------+
   +--->|  ipr  |--------->|              |       +-------+          +---------------+               |    |
   |    +-------+          +--------------+                                        ^                 |    |
   |                                                                               |                 |    |
   |                                                                               +-----------------+    |
   |                                                                                 feedback_signals     |
   |                                                                                                      |
   +------------------------------------------------------------------------------------------------------+
                                                                                            
```

Реализован в классе `ControlUnit`.

Особенности:
  * hardwired (реализован полностью на Python, для каждого типа инструкции можно однозначно 
    определить сигналы и их порядок для каждого такта исполнения инструкции)
  * step counter - необходим для много-тактовых инструкций

Сигналы:
  * `read_instruction_memory` - чтение данных по адресу `ipr` в `inr` (`inr <- mem[ipr]`)
  * `latch_inr` - записать значение из памяти инструкций в `inr`
  * signals - управляющие сигналы в DataPath
  * feedback_signals - сигналы обратной связи (`flr`)

Шаг декодирования и исполнения одной инструкции выполняется в функции `execute_next_instruction`.
Во время шага в журнал моделирования записывается исполняемая инструкция и состояние модели на конец исполнения.

Цикл симуляции осуществляется в функции simulation.

Остановка моделирования осуществляется при превышении лимита количества выполняемых инструкций, 
при отсутствии данных для чтения из порта ввода, при выполнении инструкции halt.


### Тестирование

Для тестирования выбраны 7 алгоритмов:
1. [hello](./golden/hello.yml) ([source](./examples/hello))
2. [echo](./golden/echo.yml) ([source](./examples/echo))
3. [hello_user_name](./golden/hello_user_name.yml) ([source](./examples/hello_user_name))
4. [prob5](./golden/prob5.yml) ([source](./examples/prob5))
5. [if](./golden/if.yml)
6. [math](./golden/math.yml)
7. [func](./golden/func.yml)

Golden-тесты реализованы в [integration_test.py](integration_test.py), конфигурация к ним 
находится в директории [golden](./golden).

CI:

```yaml
lint:
  stage: test
  image:
    name: ryukzak/python-tools
    entrypoint: [""]
  script:
    - poetry install
    - coverage run -m pytest --verbose
    - find . -type f -name "*.py" | xargs -t coverage report
    - ruff format --diff .
    - ruff check .
```

где:
* `ryukzak/python-tools` - docker образ содержит все необходимые для проверки утилиты
* `poetry` - управления зависимостями
* `coverage` - формирование отчёта об уровне покрытия исходного кода
* `pytest` - утилита для запуска тестов
* `ruff` - утилита для форматирования и проверки стиля кодирования

Пример использования и журнал работы процессора на примере `hello`:

```bash
% cat ./examples/hello
(prints "Hello, world!")
% cat ./examples/input/hello
% python3 translator.py ./examples/hello -v   
INFO:root:LoC: 1 code byte: 202 code instr: 20 debug lines: 29
% cat debug.txt 
##### Data memory #####
<address>      <length>       <data>
0x000          14             'Hello, world!'

##### Instruction memory #####
<address>      <hexcode>      <mnemonica>
#:
0x00000000     0x0c200011     br *0x11            (start)
prints:
0x00000001     0x06000000     push
0x00000002     0x02100000     ld #0x0
0x00000003     0x06000000     push
0x00000004     0x02500001     ld **spr+0x1
0x00000005     0x09100000     cmp #0x0
0x00000006     0x0a300008     bre *ipr+0x8
0x00000007     0x032015b4     st *0x15b4
0x00000008     0x0d400001     inc *spr+0x1
0x00000009     0x0d400000     inc *spr
0x0000000a     0x02400000     ld *spr
0x0000000b     0x09100080     cmp #0x80
0x0000000c     0x0a300002     bre *ipr+0x2
0x0000000d     0x0c3ffff7     br *ipr-0x9
0x0000000e     0x07000000     pop
0x0000000f     0x08000000     popn
0x00000010     0x05000000     ret
start:
0x00000011     0x02100000     ld #0x0             (Hello, world!)
0x00000012     0x04200001     call *0x1           (prints)
0x00000013     0x01000000     halt
% python3 machine.py target -v
DEBUG:root:br *0x11
DEBUG:root:tick=1 acr=0x0 ipr=0x11 inr=0xc200011 adr=0x0 dar=0x0 spr=0x1fff flr=0x0 stack_top=?
DEBUG:root:ld #0x0
DEBUG:root:tick=3 acr=0x0 ipr=0x12 inr=0x2100000 adr=0x0 dar=0x0 spr=0x1fff flr=0x0 stack_top=?
DEBUG:root:call *0x1
DEBUG:root:tick=7 acr=0x0 ipr=0x1 inr=0x4200001 adr=0x1ffe dar=0x13 spr=0x1ffe flr=0x0 stack_top=0x13
DEBUG:root:push
DEBUG:root:tick=10 acr=0x0 ipr=0x2 inr=0x6000000 adr=0x1ffd dar=0x0 spr=0x1ffd flr=0x0 stack_top=0x0
DEBUG:root:ld #0x0
DEBUG:root:tick=12 acr=0x0 ipr=0x3 inr=0x2100000 adr=0x1ffd dar=0x0 spr=0x1ffd flr=0x0 stack_top=0x0
DEBUG:root:push
DEBUG:root:tick=15 acr=0x0 ipr=0x4 inr=0x6000000 adr=0x1ffc dar=0x0 spr=0x1ffc flr=0x0 stack_top=0x0
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=20 acr=0x48 ipr=0x5 inr=0x2500001 adr=0x0 dar=0x48 spr=0x1ffc flr=0x0 stack_top=0x0
DEBUG:root:cmp #0x0
DEBUG:root:tick=22 acr=0x48 ipr=0x6 inr=0x9100000 adr=0x0 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x0
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=23 acr=0x48 ipr=0x7 inr=0xa300008 adr=0x0 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x0
DEBUG:root:st *0x15b4
DEBUG:root:output: '' <- 'H'
DEBUG:root:tick=26 acr=0x48 ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x48 spr=0x1ffc flr=0x1 stack_top=0x0
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=30 acr=0x48 ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0x1 spr=0x1ffc flr=0x1 stack_top=0x0
DEBUG:root:inc *spr
DEBUG:root:tick=34 acr=0x48 ipr=0xa inr=0xd400000 adr=0x1ffc dar=0x1 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:ld *spr
DEBUG:root:tick=37 acr=0x1 ipr=0xb inr=0x2400000 adr=0x1ffc dar=0x1 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:cmp #0x80
DEBUG:root:tick=39 acr=0x1 ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x1
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=40 acr=0x1 ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x1
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=42 acr=0x1 ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x1
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=47 acr=0x65 ipr=0x5 inr=0x2500001 adr=0x1 dar=0x65 spr=0x1ffc flr=0x8 stack_top=0x1
DEBUG:root:cmp #0x0
DEBUG:root:tick=49 acr=0x65 ipr=0x6 inr=0x9100000 adr=0x1 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=50 acr=0x65 ipr=0x7 inr=0xa300008 adr=0x1 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:st *0x15b4
DEBUG:root:output: 'H' <- 'e'
DEBUG:root:tick=53 acr=0x65 ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x65 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=57 acr=0x65 ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0x2 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:inc *spr
DEBUG:root:tick=61 acr=0x65 ipr=0xa inr=0xd400000 adr=0x1ffc dar=0x2 spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:ld *spr
DEBUG:root:tick=64 acr=0x2 ipr=0xb inr=0x2400000 adr=0x1ffc dar=0x2 spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:cmp #0x80
DEBUG:root:tick=66 acr=0x2 ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x2
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=67 acr=0x2 ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x2
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=69 acr=0x2 ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x2
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=74 acr=0x6c ipr=0x5 inr=0x2500001 adr=0x2 dar=0x6c spr=0x1ffc flr=0x8 stack_top=0x2
DEBUG:root:cmp #0x0
DEBUG:root:tick=76 acr=0x6c ipr=0x6 inr=0x9100000 adr=0x2 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=77 acr=0x6c ipr=0x7 inr=0xa300008 adr=0x2 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:st *0x15b4
DEBUG:root:output: 'He' <- 'l'
DEBUG:root:tick=80 acr=0x6c ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x6c spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=84 acr=0x6c ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0x3 spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:inc *spr
DEBUG:root:tick=88 acr=0x6c ipr=0xa inr=0xd400000 adr=0x1ffc dar=0x3 spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:ld *spr
DEBUG:root:tick=91 acr=0x3 ipr=0xb inr=0x2400000 adr=0x1ffc dar=0x3 spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:cmp #0x80
DEBUG:root:tick=93 acr=0x3 ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x3
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=94 acr=0x3 ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x3
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=96 acr=0x3 ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x3
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=101 acr=0x6c ipr=0x5 inr=0x2500001 adr=0x3 dar=0x6c spr=0x1ffc flr=0x8 stack_top=0x3
DEBUG:root:cmp #0x0
DEBUG:root:tick=103 acr=0x6c ipr=0x6 inr=0x9100000 adr=0x3 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=104 acr=0x6c ipr=0x7 inr=0xa300008 adr=0x3 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hel' <- 'l'
DEBUG:root:tick=107 acr=0x6c ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x6c spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=111 acr=0x6c ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0x4 spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:inc *spr
DEBUG:root:tick=115 acr=0x6c ipr=0xa inr=0xd400000 adr=0x1ffc dar=0x4 spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:ld *spr
DEBUG:root:tick=118 acr=0x4 ipr=0xb inr=0x2400000 adr=0x1ffc dar=0x4 spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:cmp #0x80
DEBUG:root:tick=120 acr=0x4 ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x4
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=121 acr=0x4 ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x4
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=123 acr=0x4 ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x4
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=128 acr=0x6f ipr=0x5 inr=0x2500001 adr=0x4 dar=0x6f spr=0x1ffc flr=0x8 stack_top=0x4
DEBUG:root:cmp #0x0
DEBUG:root:tick=130 acr=0x6f ipr=0x6 inr=0x9100000 adr=0x4 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=131 acr=0x6f ipr=0x7 inr=0xa300008 adr=0x4 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hell' <- 'o'
DEBUG:root:tick=134 acr=0x6f ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x6f spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=138 acr=0x6f ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0x5 spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:inc *spr
DEBUG:root:tick=142 acr=0x6f ipr=0xa inr=0xd400000 adr=0x1ffc dar=0x5 spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:ld *spr
DEBUG:root:tick=145 acr=0x5 ipr=0xb inr=0x2400000 adr=0x1ffc dar=0x5 spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:cmp #0x80
DEBUG:root:tick=147 acr=0x5 ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x5
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=148 acr=0x5 ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x5
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=150 acr=0x5 ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x5
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=155 acr=0x2c ipr=0x5 inr=0x2500001 adr=0x5 dar=0x2c spr=0x1ffc flr=0x8 stack_top=0x5
DEBUG:root:cmp #0x0
DEBUG:root:tick=157 acr=0x2c ipr=0x6 inr=0x9100000 adr=0x5 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=158 acr=0x2c ipr=0x7 inr=0xa300008 adr=0x5 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello' <- ','
DEBUG:root:tick=161 acr=0x2c ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x2c spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=165 acr=0x2c ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0x6 spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:inc *spr
DEBUG:root:tick=169 acr=0x2c ipr=0xa inr=0xd400000 adr=0x1ffc dar=0x6 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:ld *spr
DEBUG:root:tick=172 acr=0x6 ipr=0xb inr=0x2400000 adr=0x1ffc dar=0x6 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:cmp #0x80
DEBUG:root:tick=174 acr=0x6 ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x6
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=175 acr=0x6 ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x6
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=177 acr=0x6 ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x6
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=182 acr=0x20 ipr=0x5 inr=0x2500001 adr=0x6 dar=0x20 spr=0x1ffc flr=0x8 stack_top=0x6
DEBUG:root:cmp #0x0
DEBUG:root:tick=184 acr=0x20 ipr=0x6 inr=0x9100000 adr=0x6 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=185 acr=0x20 ipr=0x7 inr=0xa300008 adr=0x6 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello,' <- ' '
DEBUG:root:tick=188 acr=0x20 ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x20 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=192 acr=0x20 ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0x7 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:inc *spr
DEBUG:root:tick=196 acr=0x20 ipr=0xa inr=0xd400000 adr=0x1ffc dar=0x7 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:ld *spr
DEBUG:root:tick=199 acr=0x7 ipr=0xb inr=0x2400000 adr=0x1ffc dar=0x7 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:cmp #0x80
DEBUG:root:tick=201 acr=0x7 ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x7
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=202 acr=0x7 ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x7
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=204 acr=0x7 ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x7
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=209 acr=0x77 ipr=0x5 inr=0x2500001 adr=0x7 dar=0x77 spr=0x1ffc flr=0x8 stack_top=0x7
DEBUG:root:cmp #0x0
DEBUG:root:tick=211 acr=0x77 ipr=0x6 inr=0x9100000 adr=0x7 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=212 acr=0x77 ipr=0x7 inr=0xa300008 adr=0x7 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, ' <- 'w'
DEBUG:root:tick=215 acr=0x77 ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x77 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=219 acr=0x77 ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0x8 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:inc *spr
DEBUG:root:tick=223 acr=0x77 ipr=0xa inr=0xd400000 adr=0x1ffc dar=0x8 spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:ld *spr
DEBUG:root:tick=226 acr=0x8 ipr=0xb inr=0x2400000 adr=0x1ffc dar=0x8 spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:cmp #0x80
DEBUG:root:tick=228 acr=0x8 ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x8
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=229 acr=0x8 ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x8
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=231 acr=0x8 ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x8
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=236 acr=0x6f ipr=0x5 inr=0x2500001 adr=0x8 dar=0x6f spr=0x1ffc flr=0x8 stack_top=0x8
DEBUG:root:cmp #0x0
DEBUG:root:tick=238 acr=0x6f ipr=0x6 inr=0x9100000 adr=0x8 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=239 acr=0x6f ipr=0x7 inr=0xa300008 adr=0x8 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, w' <- 'o'
DEBUG:root:tick=242 acr=0x6f ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x6f spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=246 acr=0x6f ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0x9 spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:inc *spr
DEBUG:root:tick=250 acr=0x6f ipr=0xa inr=0xd400000 adr=0x1ffc dar=0x9 spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:ld *spr
DEBUG:root:tick=253 acr=0x9 ipr=0xb inr=0x2400000 adr=0x1ffc dar=0x9 spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:cmp #0x80
DEBUG:root:tick=255 acr=0x9 ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x9
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=256 acr=0x9 ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x9
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=258 acr=0x9 ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0x9
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=263 acr=0x72 ipr=0x5 inr=0x2500001 adr=0x9 dar=0x72 spr=0x1ffc flr=0x8 stack_top=0x9
DEBUG:root:cmp #0x0
DEBUG:root:tick=265 acr=0x72 ipr=0x6 inr=0x9100000 adr=0x9 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=266 acr=0x72 ipr=0x7 inr=0xa300008 adr=0x9 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, wo' <- 'r'
DEBUG:root:tick=269 acr=0x72 ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x72 spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=273 acr=0x72 ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0xa spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:inc *spr
DEBUG:root:tick=277 acr=0x72 ipr=0xa inr=0xd400000 adr=0x1ffc dar=0xa spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:ld *spr
DEBUG:root:tick=280 acr=0xa ipr=0xb inr=0x2400000 adr=0x1ffc dar=0xa spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:cmp #0x80
DEBUG:root:tick=282 acr=0xa ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xa
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=283 acr=0xa ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xa
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=285 acr=0xa ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xa
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=290 acr=0x6c ipr=0x5 inr=0x2500001 adr=0xa dar=0x6c spr=0x1ffc flr=0x8 stack_top=0xa
DEBUG:root:cmp #0x0
DEBUG:root:tick=292 acr=0x6c ipr=0x6 inr=0x9100000 adr=0xa dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=293 acr=0x6c ipr=0x7 inr=0xa300008 adr=0xa dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, wor' <- 'l'
DEBUG:root:tick=296 acr=0x6c ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x6c spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=300 acr=0x6c ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0xb spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:inc *spr
DEBUG:root:tick=304 acr=0x6c ipr=0xa inr=0xd400000 adr=0x1ffc dar=0xb spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:ld *spr
DEBUG:root:tick=307 acr=0xb ipr=0xb inr=0x2400000 adr=0x1ffc dar=0xb spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:cmp #0x80
DEBUG:root:tick=309 acr=0xb ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xb
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=310 acr=0xb ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xb
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=312 acr=0xb ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xb
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=317 acr=0x64 ipr=0x5 inr=0x2500001 adr=0xb dar=0x64 spr=0x1ffc flr=0x8 stack_top=0xb
DEBUG:root:cmp #0x0
DEBUG:root:tick=319 acr=0x64 ipr=0x6 inr=0x9100000 adr=0xb dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=320 acr=0x64 ipr=0x7 inr=0xa300008 adr=0xb dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, worl' <- 'd'
DEBUG:root:tick=323 acr=0x64 ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x64 spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=327 acr=0x64 ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0xc spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:inc *spr
DEBUG:root:tick=331 acr=0x64 ipr=0xa inr=0xd400000 adr=0x1ffc dar=0xc spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:ld *spr
DEBUG:root:tick=334 acr=0xc ipr=0xb inr=0x2400000 adr=0x1ffc dar=0xc spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:cmp #0x80
DEBUG:root:tick=336 acr=0xc ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xc
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=337 acr=0xc ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xc
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=339 acr=0xc ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xc
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=344 acr=0x21 ipr=0x5 inr=0x2500001 adr=0xc dar=0x21 spr=0x1ffc flr=0x8 stack_top=0xc
DEBUG:root:cmp #0x0
DEBUG:root:tick=346 acr=0x21 ipr=0x6 inr=0x9100000 adr=0xc dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=347 acr=0x21 ipr=0x7 inr=0xa300008 adr=0xc dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, world' <- '!'
DEBUG:root:tick=350 acr=0x21 ipr=0x8 inr=0x32015b4 adr=0x15b4 dar=0x21 spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:inc *spr+0x1
DEBUG:root:tick=354 acr=0x21 ipr=0x9 inr=0xd400001 adr=0x1ffd dar=0xd spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:inc *spr
DEBUG:root:tick=358 acr=0x21 ipr=0xa inr=0xd400000 adr=0x1ffc dar=0xd spr=0x1ffc flr=0x1 stack_top=0xd
DEBUG:root:ld *spr
DEBUG:root:tick=361 acr=0xd ipr=0xb inr=0x2400000 adr=0x1ffc dar=0xd spr=0x1ffc flr=0x1 stack_top=0xd
DEBUG:root:cmp #0x80
DEBUG:root:tick=363 acr=0xd ipr=0xc inr=0x9100080 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xd
DEBUG:root:bre *ipr+0x2
DEBUG:root:tick=364 acr=0xd ipr=0xd inr=0xa300002 adr=0x1ffc dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xd
DEBUG:root:br *ipr+0xffff7
DEBUG:root:tick=366 acr=0xd ipr=0x4 inr=0xc3ffff7 adr=0xd dar=0x80 spr=0x1ffc flr=0x8 stack_top=0xd
DEBUG:root:ld **spr+0x1
DEBUG:root:tick=371 acr=0x0 ipr=0x5 inr=0x2500001 adr=0xd dar=0x0 spr=0x1ffc flr=0x8 stack_top=0xd
DEBUG:root:cmp #0x0
DEBUG:root:tick=373 acr=0x0 ipr=0x6 inr=0x9100000 adr=0xd dar=0x0 spr=0x1ffc flr=0x5 stack_top=0xd
DEBUG:root:bre *ipr+0x8
DEBUG:root:tick=375 acr=0x0 ipr=0xe inr=0xa300008 adr=0x6 dar=0x0 spr=0x1ffc flr=0x5 stack_top=0xd
DEBUG:root:pop
DEBUG:root:tick=379 acr=0xd ipr=0xf inr=0x7000000 adr=0x1ffc dar=0xd spr=0x1ffd flr=0x5 stack_top=0xd
DEBUG:root:popn
DEBUG:root:tick=380 acr=0xd ipr=0x10 inr=0x8000000 adr=0x1ffc dar=0xd spr=0x1ffe flr=0x5 stack_top=0x13
DEBUG:root:ret
DEBUG:root:tick=384 acr=0xd ipr=0x13 inr=0x5000000 adr=0x1ffe dar=0x13 spr=0x1fff flr=0x5 stack_top=?
DEBUG:root:halt
INFO:root:instr: 142 ticks: 384
Hello, world!
```

Пример проверки исходного кода:

```bash
% poetry run pytest . -v
============================================================================================= test session starts ==============================================================================================
platform darwin -- Python 3.12.0, pytest-7.4.3, pluggy-1.3.0 -- /Users/andryssssss/Library/Caches/pypoetry/virtualenvs/lispfuck-qe1KGZJD-py3.12/bin/python
cachedir: .pytest_cache
rootdir: /Users/andryssssss/Programming/Projects/Python/ITMO/ComputerArchitectureLab3
configfile: pyproject.toml
plugins: golden-0.2.2
collected 7 items                                                                                                                                                                                              

integration_test.py::test_translator_and_machine[golden/func.yml] PASSED                                                                                                                                 [ 14%]
integration_test.py::test_translator_and_machine[golden/if.yml] PASSED                                                                                                                                   [ 28%]
integration_test.py::test_translator_and_machine[golden/math.yml] PASSED                                                                                                                                 [ 42%]
integration_test.py::test_translator_and_machine[golden/hello.yml] PASSED                                                                                                                                [ 57%]
integration_test.py::test_translator_and_machine[golden/prob5.yml] PASSED                                                                                                                                [ 71%]
integration_test.py::test_translator_and_machine[golden/hello_user_name.yml] PASSED                                                                                                                      [ 85%]
integration_test.py::test_translator_and_machine[golden/echo.yml] PASSED                                                                                                                                 [100%]

============================================================================================== 7 passed in 0.61s ===============================================================================================
% poetry run ruff check .
% poetry run ruff format .
5 files left unchanged
```

Статистика:

| alg             | LoC | code byte | code instr | debug lines | instr | ticks |
|-----------------|-----|-----------|------------|-------------|-------|-------|
| hello           | 1   | 202       | 20         | 29          | 142   | 384   |
| echo            | 1   | 174       | 41         | 50          | 126   | 348   |
| hello_user_name | 4   | 278       | 47         | 58          | 250   | 688   |
| prob5           | 16  | 698       | 132        | 145         | 2781  | 7266  |
| if              | 1   | 206       | 49         | 58          | 174   | 484   |
| math            | 6   | 550       | 131        | 143         | 191   | 534   |
| func            | 20  | 562       | 120        | 134         | 275   | 756   |
