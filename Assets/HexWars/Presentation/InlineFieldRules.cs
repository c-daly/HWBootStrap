using System.Globalization;

namespace HexWars.Presentation
{
    public static class InlineFieldRules
    {
        public static bool TryInt(
            string text,
            int min,
            int max,
            bool blankMeansZero,
            out int value,
            out string error)
        {
            text = (text ?? string.Empty).Trim();
            if (text.Length == 0 && blankMeansZero)
                text = "0";

            if (text.Length == 0)
            {
                value = 0;
                error = "Required";
                return false;
            }

            if (!int.TryParse(
                    text,
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out value))
            {
                error = "Enter a whole number";
                return false;
            }

            if (value < min || value > max)
            {
                error = max == int.MaxValue
                    ? $"Use {min} or more"
                    : $"Use {min}\u2013{max}";
                return false;
            }

            error = string.Empty;
            return true;
        }
    }
}
