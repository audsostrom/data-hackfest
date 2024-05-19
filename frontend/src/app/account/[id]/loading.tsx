import {CircularProgress} from "@mui/material";
import Container from "@mui/material/Container";

export default function AccountLoading() {
  return (
    <Container sx={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      width: '100%',
    }}>
      <CircularProgress />
    </Container>
  );
}