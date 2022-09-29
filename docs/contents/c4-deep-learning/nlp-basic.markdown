# Python Text Basic

## Working with text file

In this section we'll cover

* Working with f-strings (formatted string literals) to format printed text

* Working with Files - opening, reading, writing and appending text files

### Formatted String Literals (f-strings)

```python
name = 'Fred'
# Using the old .format() method:
print('His name is {var}.'.format(var=name))
# Using f-strings:
print(f'His name is {name}.')
```

Pass `!r` to get the <strong>string representation</strong>:

```python
print(f'His name is {name!r}')
---result---
His name is 'Fred'
```

#### Quotation marks

```python
d = {'a':123,'b':456}
print(f'Address: {d['a']} Main Street')
---result---
SyntaxError: f-string: unmatched '['

d = {'a':123,'b':456}
print(f"Address: {d['a']} Main Street")
---result---
Address: 123 Main Street
```

####  Minimum Widths, Alignment and Padding

You can pass arguments inside a nested set of curly braces to set a minimum width for the field, the alignment and even padding characters.

```python
library = [('Author', 'Topic', 'Pages'), ('Twain', 'Rafting', 601), ('Feynman', 'Physics', 95), ('Hamilton', 'Mythology', 144)]
for book in library:
    print(f'{book[0]:{10}} {book[1]:{8}} {book[2]:{7}}')
---result---
Author     Topic    Pages  
Twain      Rafting      601
Feynman    Physics       95
Hamilton   Mythology     14411111
```

Here the first three lines align, except `Pages` follows a default left-alignment while numbers are right-aligned. Also, the fourth line's page number is pushed to the right as `Mythology` exceeds the minimum field width of `8`. When setting minimum field widths make sure to take the longest item into account.

To set the alignment, use the character `<` for left-align,  `^` for center, `>` for right.<br>To set padding, precede the alignment character with the padding character (`-` and `.` are common choices).

Let's make some adjustments:

```python
for book in library:
    print(f'{book[0]:{10}} {book[1]:{10}} {book[2]:.>{7}}') # here .> was added
---result---
Author     Topic      ..Pages
Twain      Rafting    ....601
Feynman    Physics    .....95
Hamilton   Mythology  ....144
```

#### Date formatting

```python
from datetime import datetime
today = datetime(year=2018, month=1, day=27)
print(f'{today:%B %d, %Y}')
---result---
January 27, 2018
```

For more info on formatted string literals visit https://docs.python.org/3/reference/lexical_analysis.html#f-strings

- `%d`: Returns the **day** of the month, from 1 to 31.
- `%m`: Returns the **month** of the year, from 1 to 12.
- `%Y`: Returns the year in four-digit format (**Year** with century). like, 2021.
- `%y`: Reurns year in two-digit format (**year** without century). like, 19, 20, 21
- `%A`: Returns the full name of the **weekday**. Like, Monday, Tuesday
- `%a`: Returns the short name of the **weekday** (First three character.). Like, Mon, Tue
- `%B`: Returns the full name of the **month**. Like, June, March
- `%b`: Returns the short name of the **month** (First three character.). Like, Mar, Jun
- `%H`: Returns the **hour**. from 01 to 23.
- `%I`: Returns the **hour** in 12-hours format. from 01 to 12.
- `%M`: Returns the **minute**, from 00 to 59.
- `%S`: Returns the **second**, from 00 to 59.
- `%f`: Return the **microseconds** from 000000 to 999999
- `%p`: Return time in **AM/PM** format
- `%c`: Returns a **locale’s appropriate date and time** representation
- `%x`: Returns a locale’s appropriate date representation
- `%X`: Returns a locale’s appropriate time representation
- `%z`: Return the **UTC offset** in the form `±HHMM[SS[.ffffff]]` (empty string if the object is naive).
- `%Z`: Return the **Time zone name** (empty string if the object is naive).
- `%j`: Returns the day of the year from *01 to 366*
- `%w`: Returns weekday as a decimal number, where 0 is Sunday and 6 is Saturday.
- `%U`: Returns the week number of the year (Sunday as the first day of the week) from 00 to 53
- `%W`: Returns the week number of the year (Monday as the first day of the week) from 00 to 53

## Working with PDF

### Working with PyPDF2

```python
# note the capitalization
import PyPDF2
```

#### Reading PDFs

First we open a pdf, then create a reader object for it. Notice how we use the binary method of reading , 'rb', instead of just 'r'.

```python
# Notice we read it as a binary with 'rb'
f = open('US_Declaration.pdf','rb')
pdf_reader = PyPDF2.PdfFileReader(f)

pdf_reader.numPages
# 5

page_one = pdf_reader.getPage(0)
page_one_text = page_one.extractText()
page_one_text
# "Declaration of Independence\nIN CONGRESS, July 4, 1776.  The unanimous Declaration ....

f.close()
```

#### Adding to PDFs

We can not write to PDFs using Python because of the differences between the single string type of Python, and the variety of fonts, placements, and other parameters that a PDF could have.

What we **can** do is copy pages and append pages to the end.

```python
f = open('US_Declaration.pdf','rb')
pdf_reader = PyPDF2.PdfFileReader(f)
first_page = pdf_reader.getPage(0)
pdf_writer = PyPDF2.PdfFileWriter()
pdf_writer.addPage(first_page)
pdf_output = open("Some_New_Doc.pdf","wb")
pdf_writer.write(pdf_output)
pdf_output.close()
f.close()

# Now we have copied a page and added it to another new document!
```

#### Simple example

Let's try to grab all the text from this PDF file:

```python
f = open('US_Declaration.pdf','rb')
# List of every page's text.
# The index will correspond to the page number.
pdf_text = [0]  # zero is a placehoder to make page 1 = index 1
pdf_reader = PyPDF2.PdfFileReader(f)
for p in range(pdf_reader.numPages):
    page = pdf_reader.getPage(p)
    pdf_text.append(page.extractText())
f.close()
```

## Regular Expressions

> A regular expression (shortened as regex [...]) is a sequence of characters that specifies a search pattern in text. [...] used by string-searching algorithms for "find" or "find and replace" operations on strings, or for input validation.

1. Import the regex module with `import re`.
2. Create a Regex object with the `re.compile()` function. (Remember to use a raw string.)
3. Pass the string you want to search into the Regex object’s `search()` method. This returns a `Match` object.
4. Call the Match object’s `group()` method to return a string of the actual matched text.

### Regex symbols

| Symbol                   | Matches                                                |
| ------------------------ | ------------------------------------------------------ |
| `?`                      | zero or one of the preceding group.                    |
| `*`                      | zero or more of the preceding group.                   |
| `+`                      | one or more of the preceding group.                    |
| `{n}`                    | exactly n of the preceding group.                      |
| `{n,}`                   | n or more of the preceding group.                      |
| `{,m}`                   | 0 to m of the preceding group.                         |
| `{n,m}`                  | at least n and at most m of the preceding p.           |
| `{n,m}?` or `*?` or `+?` | performs a non-greedy match of the preceding p.        |
| `^spam`                  | means the string must begin with spam.                 |
| `spam$`                  | means the string must end with spam.                   |
| `.`                      | any character, except newline characters.              |
| `\d`, `\w`, and `\s`     | a digit, word, or space character, respectively.       |
| `\D`, `\W`, and `\S`     | anything except a digit, word, or space, respectively. |
| `[abc]`                  | any character between the brackets (such as a, b, ).   |
| `[^abc]`                 | any character that isn’t between the brackets.         |

### Matching regex objects

```python
import re
phone_regex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
text = 'My phone number is 093-422-0591'
match = phone_regex.search(text)
match
# <re.Match object; span=(19, 31), match='093-422-0591'>
print(f'Phone number found: {match.group()}')
# Phone number found: 093-422-0591
```



### Grouping with parentheses (Dấu ngoặc đơn)

```python
phone_num_regex = re.compile(r'(\d\d\d)-(\d\d\d-\d\d\d\d)')
mo = phone_num_regex.search('My number is 415-555-4242.')

mo.group(1)
# '415'

mo.group(2)
# '555-4242'

mo.group(0)
# '415-555-4242'

mo.group()
# '415-555-4242'

mo.groups()
('415', '555-4242')

area_code, main_number = mo.groups()

print(area_code)
415

print(main_number)
555-4242
```

https://www.pythoncheatsheet.org/cheatsheet/regular-expressions
