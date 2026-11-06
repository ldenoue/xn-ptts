const CONSONANTS = 'B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH'.split(' ');
const VOWELS = 'AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW'.split(' ');
export const SPECIAL = ['<pad>', '<bos>', '<eos>', '<space>', '.', ',', '?', '!', ';', ':'];
export const SYMBOLS = [...SPECIAL, ...CONSONANTS, ...VOWELS.flatMap((vowel) => ['0', '1', '2'].map((stress) => vowel + stress))];
const SYMBOL_TO_ID = new Map(SYMBOLS.map((symbol, index) => [symbol, index]));
const ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(' ');
const TENS = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split(' ');
const ORDINALS = { one: 'first', two: 'second', three: 'third', five: 'fifth', eight: 'eighth', nine: 'ninth', twelve: 'twelfth' };
const MONTHS = 'January February March April May June July August September October November December'.split(' ');

let lexiconPromise;
const LEXICON_URL = new URL('./pivo-model/cmudict.dict', import.meta.url);

function cardinal(number) {
  if (number < 0) return `minus ${cardinal(-number)}`;
  if (number < 20) return ONES[number];
  if (number < 100) return TENS[Math.floor(number / 10)] + (number % 10 ? ` ${cardinal(number % 10)}` : '');
  for (const [scale, label] of [[1e12, 'trillion'], [1e9, 'billion'], [1e6, 'million'], [1000, 'thousand'], [100, 'hundred']]) {
    if (number >= scale) {
      if (number >= 1e15) return String(number).split('').map((digit) => ONES[Number(digit)]).join(' ');
      const remainder = number % scale;
      return `${cardinal(Math.floor(number / scale))} ${label}${remainder ? ` ${cardinal(remainder)}` : ''}`;
    }
  }
  throw new Error('Could not read that number.');
}

function ordinal(number) {
  const words = cardinal(number).split(' ');
  const last = words.at(-1);
  words[words.length - 1] = ORDINALS[last] ?? (last.endsWith('y') ? `${last.slice(0, -1)}ieth` : `${last}th`);
  return words.join(' ');
}

function normalize(raw) {
  let value = raw.normalize('NFKC')
    .replace(/[’‘]/g, "'").replace(/[“”]/g, '"').replace(/[—–]/g, ',').replace(/…/g, '...');

  value = value.replace(/\b(\d{4})-(\d{2})-(\d{2})\b/g, (_, y, m, d) => {
    const year = Number(y); const month = Number(m); const day = Number(d);
    const date = new Date(Date.UTC(year, month - 1, day));
    if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day) throw new Error('That ISO date is invalid.');
    return `${MONTHS[month - 1]} ${ordinal(day)}, ${cardinal(year)}`;
  });
  if (/\d\/\d/.test(value)) throw new Error('Write slash dates or fractions in words, or use an ISO date.');

  const abbreviations = { Mr: 'Mister', Mrs: 'Missus', Dr: 'Doctor', Prof: 'Professor', St: 'Saint', vs: 'versus', etc: 'et cetera' };
  for (const [short, long] of Object.entries(abbreviations)) value = value.replace(new RegExp(`\\b${short}\\.(?=\\s|$)`, 'gi'), long);
  if (/[$£€]\d[\d,]*\.\d{3,}/.test(value)) throw new Error('Currency amounts must have at most two decimal places.');
  value = value.replace(/([$£€])(\d[\d,]*(?:\.\d{1,2})?)(?!\d)/g, (_, symbol, rawNumber) => {
    const [wholeText, fraction = ''] = rawNumber.replaceAll(',', '').split('.');
    const whole = Number(wholeText); const cents = Number(fraction.padEnd(2, '0'));
    const unit = { '$': 'dollar', '£': 'pound', '€': 'euro' }[symbol];
    let result = `${cardinal(whole)} ${unit}${whole === 1 ? '' : 's'}`;
    if (cents) result += ` and ${cardinal(cents)} ${symbol === '£' ? (cents === 1 ? 'penny' : 'pence') : (cents === 1 ? 'cent' : 'cents')}`;
    return result;
  });
  value = value.replace(/\b(\d+)(?:st|nd|rd|th)\b/gi, (_, number) => ordinal(Number(number)));
  value = value.replace(/\b\d{1,3}(?:,\d{3})+\b/g, (match) => match.replaceAll(',', ''));
  value = value.replace(/-?(?:\d+\.\d+|\d+)/g, (match) => {
    const negative = match.startsWith('-'); const unsigned = match.replace('-', ''); const [whole, fraction] = unsigned.split('.');
    return `${negative ? 'minus ' : ''}${cardinal(Number(whole))}${fraction ? ` point ${fraction.split('').map((digit) => ONES[Number(digit)]).join(' ')}` : ''}`;
  });
  value = value.replaceAll('%', ' percent ').replaceAll('&', ' and ').replaceAll('+', ' plus ');
  value = value.replace(/\bNASA\b/g, 'nasa').replace(/\b[A-Z]{2,5}\b/g, (match) => match.split('').join(' '));
  value = value.replaceAll('-', ' ').replace(/[()\[\]{}/"]/g, ' ');
  const unsupported = [...new Set(value.match(/[^a-zA-Z\s'.,?!;:]/g) ?? [])];
  if (unsupported.length) throw new Error(`Unsupported character${unsupported.length > 1 ? 's' : ''}: ${unsupported.join(' ')}`);
  return value.replace(/\s+/g, ' ').trim();
}

async function lexicon() {
  if (!lexiconPromise) lexiconPromise = fetch(LEXICON_URL).then((response) => {
    if (!response.ok) throw new Error('Could not load the pronunciation dictionary.');
    return response.text();
  }).then((raw) => {
    const dictionary = new Map();
    for (const line of raw.split('\n')) {
      const parts = line.split('#', 1)[0].trim().split(/\s+/);
      if (parts.length < 2) continue;
      const word = parts[0].replace(/\(\d+\)$/, '').toLowerCase();
      if (!dictionary.has(word)) dictionary.set(word, parts.slice(1));
    }
    return dictionary;
  });
  return lexiconPromise;
}

export async function phonemize(text) {
  const normalizedText = normalize(text);
  const dictionary = await lexicon();
  const tokens = ['<bos>'];
  const unknownWords = [];
  for (const item of normalizedText.match(/[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]/g) ?? []) {
    if (SPECIAL.includes(item)) { tokens.push(item); continue; }
    const word = item.toLowerCase();
    let phones = dictionary.get(word);
    if (!phones) {
      unknownWords.push(word);
      phones = [...word].filter((letter) => /[a-z]/.test(letter)).flatMap((letter) => dictionary.get(letter) ?? []);
    }
    if (!['<bos>', '<space>'].includes(tokens.at(-1))) tokens.push('<space>');
    for (const phone of phones) {
      if (!SYMBOL_TO_ID.has(phone)) throw new Error(`Unknown ARPAbet phone “${phone}” in “${word}”.`);
      tokens.push(phone);
    }
  }
  tokens.push('<eos>');
  return { normalizedText, tokenIds: tokens.map((token) => SYMBOL_TO_ID.get(token)), symbols: tokens, unknownWords: [...new Set(unknownWords)] };
}
